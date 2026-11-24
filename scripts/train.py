"""
Train and evaluate GeometryMMKG on one v0 dataset (RESEARCH_PLAN.md P2.1, P2.8, section 4).

  python scripts/train.py --data data/Spatial-VRD --variant full --epochs 100 --seed 42
  python scripts/train.py --data data/Spatial-VRD --variant full --rel-gate --dev     # development run

* Variants (section 3.1): full, no_depth, concat, mean, fixed, constant, constant_fixed, depth_map_only,
  depth_stats_only (core/models/end_to_end.py VARIANTS; the last two split the depth input of the scene graphs into
  the depth map and the depth statistics, VARIANT_DEPTH_STATS).
* Frozen design (P2.8 / P2.9, 19/11/2026) is the default: --input-norm, --depth-dropout 0.3, --text-proj and
  --depth-stats auto (the dataset's ent_depth_stats.pt on the scene graphs Spatial-VRD, Spatial-VRD-Trans and
  Spatial-VG; none on the knowledge graphs). Ablations switch them off: --no-input-norm, --depth-dropout 0,
  --no-text-proj, --depth-stats none. --rel-gate (tried and dropped) stays off by default.
  --depth-stats also takes data (the dataset table) or PATH[:KEY] (a scripts/vrd_depth_stats.py output, development).
  The same options can be passed to every variant; each variant uses those that apply to it
  (`effective_options`), and the options tag (folder / file names) is the same for all of them.
* Training: in-batch negatives on both sides, train-only false-negative filter, self-adversarial weighted
  margin ranking (core/models/kg_reasoning.py).
* Model selection: filtered MRR (head + tail) on valid every --valid-every epochs and after the last epoch;
  the best checkpoint is evaluated on test once. Spatial-VG validates on a fixed 5,000-triple subset (4.9),
  the same subset for every variant and seed. Each validation also logs the gate's weight on depth.
* --dev: model development (P2.8). Valid only: test is never evaluated, no ranks, no checkpoint.
* --tune: one point of the P2.10 grid (implies --dev); the stem and folder carry the configuration
  (lr<lr>-do<dropout>), default folder results/geometry-tune/<dataset>/<variant>/<config>/.
  scripts/tune_summary.py picks the best configuration per variant.
* Epoch budget (P2.10): --epochs / --valid-every default to EPOCH_BUDGET of the dataset.
* Resume (every run): after each validation the full training state (model, optimizer, RNG states, best
  checkpoint, history) is written atomically to <out>/resume_<stem>.pt; a later start with the same stem and
  the same training arguments continues from it (same batches and dropout masks as an uninterrupted run), and the
  epochs where a run was resumed are recorded in the metrics (resumed_at). The file is removed once the run has
  finished. --no-resume neither reads nor writes it.
* Output: metrics_<stem>.json, plus ranks_<stem>_test.csv (for scripts/compute_metrics.py) and ckpt_<stem>.pt
  outside --dev; stem = geometry_<dataset>_<variant>[_<options tag>]_seed<seed>.
  Default folder: results/geometry/<dataset>/<variant>/, or results/geometry-dev/<dataset>/<tag>/<variant>/.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from core.data_engineering.fast_dataset import (  # noqa: E402
    TripleBatches, get_all_true_triplets, load_depth_stats, load_entity_features, load_triples, read_count)
from core.models.end_to_end import VARIANT_DEPTH_STATS, VARIANTS, GeometryMMKGModel, effective_options  # noqa: E402
from core.models.evaluation import evaluate_link_prediction, format_metrics, save_ranks  # noqa: E402
from core.models.kg_reasoning import TripleIndex, in_batch_loss  # noqa: E402
from core.reproducibility import set_seed  # noqa: E402

VALID_SUBSET_SEED = 0       # fixed: every variant / seed validates on the same subset
# P2.10 (decided 19/11/2026): maximum epochs and validation interval per dataset; the checkpoint is chosen on valid.
EPOCH_BUDGET = {
    "FB15k-237-IMG": (200, 10),
    "FB15K-237-ind": (200, 10),
    "MKG-Y": (1000, 50),
    "Spatial-VRD": (100, 10),
    "Spatial-VRD-Trans": (100, 10),
    "Spatial-VG": (100, 10),
}
DEFAULT_BUDGET = (100, 10)
VG_VALID_MAX = 5000         # RESEARCH_PLAN.md 4.9
GATE_SAMPLE = 4096          # entities (with an image) used for the gate statistics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset folder, e.g. data/Spatial-VRD")
    p.add_argument("--variant", default="full", choices=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=None, help="default: EPOCH_BUDGET of the dataset (P2.10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--margin", type=float, default=5.0)
    p.add_argument("--adv-temp", type=float, default=1.0, help="self-adversarial temperature")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rel-gate", action="store_true", help="per-relation gate bias (P2.2; tried, not in the design)")
    p.add_argument("--input-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="separate LayerNorm on RGB and depth before fusion (design: on)")
    p.add_argument("--depth-dropout", type=float, default=0.3,
                   help="drop depth per entity while training (design: 0.3; 0 = off)")
    p.add_argument("--text-proj", action=argparse.BooleanOptionalAction, default=True,
                   help="learned projection of text, identity at init (design: on)")
    p.add_argument("--depth-stats", default="auto", metavar="auto|none|data|PATH[:KEY]",
                   help="per-entity depth statistics added as W_s s after the normalisation, W_s = 0 at init (P2.9). "
                        "auto (design) = the dataset table on scene graphs, none elsewhere; none = off; data = "
                        "<data>/ent_depth_stats.pt; or a scripts/vrd_depth_stats.py output (KEY default grid9)")
    p.add_argument("--dev", action="store_true", help="development run: valid only, never test")
    p.add_argument("--valid-every", type=int, default=None,
                   help="epochs between validations (default: EPOCH_BUDGET of the dataset)")
    p.add_argument("--tune", action="store_true", help="a P2.10 grid point: --dev plus the configuration in the names")
    p.add_argument("--valid-max", type=int, default=None,
                   help=f"validate on a fixed random subset of this size (0 = all; default {VG_VALID_MAX} for "
                        "Spatial-VG, else all)")
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--keep-on-gpu", choices=["auto", "yes", "no"], default="auto",
                   help="feature / fused tables on the GPU (auto: if < 50%% of free memory)")
    p.add_argument("--out", default=None, help="output folder (default: see above)")
    p.add_argument("--no-ckpt", action="store_true", help="do not write the best checkpoint")
    p.add_argument("--no-resume", action="store_true", help="do not read or write <out>/resume_<stem>.pt")
    return p.parse_args(argv)


def options_tag(args):
    """Folder / file tag of the requested development options ('base' = none)."""
    parts = []
    if args.rel_gate:
        parts.append("relgate")
    if args.input_norm:
        parts.append("inorm")
    if args.depth_dropout > 0:
        parts.append(f"ddrop{args.depth_dropout:g}")
    if args.text_proj:
        parts.append("tproj")
    if args.depth_stats:
        parts.append("dstat" if args.depth_stats == "data" else "dstat_" + depth_stats_name(args.depth_stats))
    return "-".join(parts) or "base"


SCENE_GRAPHS = ("Spatial-VRD", "Spatial-VRD-Trans", "Spatial-VG")


def config_tag(args, default_epochs=None):
    """Grid-point tag of a --tune run, e.g. 'lr0.0003-do0.1', plus '-ep<epochs>' when the run does not use the
    dataset's epoch budget (the convergence rerun of P2.10), so it never collides with the budget run."""
    tag = f"lr{args.lr:g}-do{args.dropout:g}"
    if default_epochs is not None and args.epochs not in (None, default_epochs):
        tag += f"-ep{args.epochs}"
    return tag


def resolve_depth_stats(spec, data_dir, dataset):
    """--depth-stats -> None (off), 'data' or a file spec. 'auto' = 'data' on the scene graphs (the table must exist),
    off on the knowledge graphs, whose entities do not share a scene (P2.9)."""
    if spec in (None, "", "none"):
        return None
    table = os.path.join(data_dir, "ent_depth_stats.pt")
    if spec == "auto":
        if dataset not in SCENE_GRAPHS:
            return None
        if not os.path.exists(table):
            raise FileNotFoundError(f"{dataset} is a scene graph but {table} is missing (data v0 has it; "
                                    "pass --depth-stats none to run without it)")
        return "data"
    if spec == "data" and not os.path.exists(table):
        raise FileNotFoundError(table)
    return spec


def depth_stats_name(spec):
    """'results/probe/vrd_depth_stats_dav2.pt:grid9' -> 'dav2_grid9' (the file stem after 'depth_stats_')."""
    path, key = (spec.rsplit(":", 1) if ":" in os.path.basename(spec) else (spec, "grid9"))
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.split("depth_stats_", 1)[-1]
    return f"{stem}_{key}"


RESUME_IGNORED_ARGS = ("data", "out", "keep_on_gpu", "eval_batch_size", "no_ckpt", "no_resume")


def resume_key(args):
    """Training arguments a resume state must match (paths that change between machines are compared by name)."""
    k = {a: v for a, v in vars(args).items() if a not in RESUME_IGNORED_ARGS}
    k["dataset"] = os.path.basename(os.path.normpath(args.data))
    if isinstance(k.get("depth_stats"), str) and k["depth_stats"] not in ("data", "none"):
        k["depth_stats"] = os.path.basename(k["depth_stats"])
    return k


def rng_state(gen):
    return {"gen": gen.get_state(), "torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def set_rng_state(gen, s):
    gen.set_state(s["gen"])
    torch.set_rng_state(s["torch"])
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(s["cuda"])


def save_resume(path, state):
    """Atomic write: a run killed while saving keeps the previous resume state."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path, args):
    """The resume state at path, or None if there is none; refuses a state written with other training arguments."""
    if args.no_resume or not os.path.exists(path):
        return None
    s = torch.load(path, map_location="cpu", weights_only=False)
    want, got = resume_key(args), s["resume_key"]
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    if diff:
        changes = ", ".join(f"{k}: {got.get(k)!r} -> {want.get(k)!r}" for k in diff)
        raise SystemExit(f"{path} was written with other training arguments ({changes}); "
                         "delete it to start again, or rerun with those arguments")
    return s


def code_commit():
    try:
        return subprocess.run(["git", "-C", CODE_DIR, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


@torch.no_grad()
def depth_stats_norm(model, feats):
    """Mean L2 norm of the depth-statistics term W_s s over a fixed sample of entities with an image
    (entity representations without it have norm 1). None without depth_stats."""
    if not getattr(model, "uses_depth_stats", False):
        return None
    ids = feats.mask.nonzero(as_tuple=True)[0]
    ids = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))[:GATE_SAMPLE].to(ids.device)]
    g = model.fusion_module.depth_stats_proj(feats.gather_geo(ids))
    return float(g.norm(dim=1).mean())


def strip_ranks(m):
    return {k: v for k, v in m.items() if k != "ranks"}


@torch.no_grad()
def gate_stats(model, feats, n_rel):
    """Weight on depth (1 - G, or 1 - a for 'fixed'), averaged over dimensions, for a fixed sample of entities
    with an image: mean, spread over entities, and (rel_gate) spread over relations. None without a gate."""
    vf = model.vision_fusion
    ids = feats.mask.nonzero(as_tuple=True)[0]
    ids = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))[:GATE_SAMPLE].to(ids.device)]
    _, rgb, depth, _ = feats.gather(ids)
    model.eval()
    if not model.relation_conditioned:
        s = vf.depth_share(rgb, depth)
        if s is None:
            return None
        if s.dim() == 0:
            return {"mean": float(s)}
        e = s.mean(dim=1)
        return {"mean": float(e.mean()), "std_entities": float(e.std())}
    per_rel = torch.stack([vf.depth_share(rgb, depth, torch.full((len(ids),), r, dtype=torch.long,
                                                                  device=rgb.device)).mean(dim=1)
                           for r in range(n_rel)])                    # [R, n]
    rel_mean = per_rel.mean(dim=1)
    return {"mean": float(per_rel.mean()), "std_entities": float(per_rel.std(dim=1).mean()),
            "std_relations": float(rel_mean.std()), "per_relation": [round(float(x), 4) for x in rel_mean]}


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.abspath(args.data)
    dataset = os.path.basename(os.path.normpath(data_dir))
    args.depth_stats = resolve_depth_stats(args.depth_stats, data_dir, dataset)
    budget_epochs, budget_valid = EPOCH_BUDGET.get(dataset, DEFAULT_BUDGET)
    args.epochs = args.epochs or budget_epochs
    args.valid_every = args.valid_every or budget_valid
    if args.tune:
        args.dev = True
    tag = options_tag(args)
    stem = (f"geometry_{dataset}_{args.variant}{'' if tag == 'base' else '_' + tag}"
            + (f"_{config_tag(args, budget_epochs)}" if args.tune else "") + f"_seed{args.seed}")
    if args.out:
        out_dir = args.out
    elif args.tune:
        out_dir = os.path.join(CODE_DIR, "results", "geometry-tune", dataset, args.variant, config_tag(args, budget_epochs))
    elif args.dev:
        out_dir = os.path.join(CODE_DIR, "results", "geometry-dev", dataset, tag, args.variant)
    else:
        out_dir = os.path.join(CODE_DIR, "results", "geometry", dataset, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    keep = {"auto": None, "yes": True, "no": False}[args.keep_on_gpu]
    valid_max = args.valid_max if args.valid_max is not None else (VG_VALID_MAX if dataset == "Spatial-VG" else 0)
    print(f"{stem}{' [dev: valid only]' if args.dev else ''} | device {device} | code {code_commit()} | "
          f"args {vars(args)}", flush=True)

    gen = set_seed(args.seed)
    n_ent = read_count(os.path.join(data_dir, "entity2id.txt"))
    n_rel = read_count(os.path.join(data_dir, "relation2id.txt"))
    feats = load_entity_features(data_dir, device, keep_on_gpu=keep)
    train = load_triples(data_dir, "train")
    valid = load_triples(data_dir, "valid")
    test = None if args.dev else load_triples(data_dir, "test")
    if valid_max and len(valid) > valid_max:
        sub = torch.randperm(len(valid), generator=torch.Generator().manual_seed(VALID_SUBSET_SEED))[:valid_max]
        valid = valid[sub.sort().values]
    all_true = get_all_true_triplets(data_dir)   # filter set = train + valid + test (section 4.3)
    train_index = TripleIndex(train, n_ent, n_rel, device=device)
    n_missing = int((~feats.mask).sum())
    depth_stats_info = None
    if args.depth_stats:
        _spec = os.path.join(data_dir, "ent_depth_stats.pt") if args.depth_stats == "data" else args.depth_stats
        feats.geo, depth_stats_info = load_depth_stats(_spec, n_ent, feats.mask, train)
        feats.geo = feats.geo.to(feats.store)
        print(f"depth statistics: {depth_stats_info}", flush=True)
    print(f"{n_ent:,} entities ({n_missing:,} without image), {n_rel} relations | train {len(train):,} | "
          f"valid {len(valid):,}{' (subset)' if valid_max else ''}"
          f"{'' if test is None else f' | test {len(test):,}'} | features stored on {feats.store}", flush=True)

    model_kwargs = dict(VARIANTS[args.variant])
    if VARIANT_DEPTH_STATS.get(args.variant) and depth_stats_info is None:
        raise SystemExit(f"variant {args.variant} needs depth statistics; {dataset} has none (or --depth-stats none)")
    options = effective_options(model_kwargs, args.rel_gate, args.input_norm, args.depth_dropout, args.text_proj,
                                0 if depth_stats_info is None else depth_stats_info["dim"],
                                VARIANT_DEPTH_STATS.get(args.variant))
    model = GeometryMMKGModel(n_rel, embed_dim=feats.text.shape[1], dropout=args.dropout,
                              **model_kwargs, **options).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"variant {args.variant} = {model_kwargs} | options {options} (tag {tag}) | {n_params:,} parameters",
          flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate(triples, desc, return_ranks=False):
        if device.type == "cuda":
            torch.cuda.empty_cache()  # release training buffers so the fused table can stay on the GPU (P1.6)
        return evaluate_link_prediction(
            model, feats.text, feats.rgb, feats.depth, TripleBatches(triples, args.eval_batch_size), all_true,
            device, keep_on_gpu=keep, return_ranks=return_ranks, desc=desc, all_mask=feats.mask,
            all_geo=feats.geo if model.uses_depth_stats else None)

    history, best = [], {"mrr": -1.0, "epoch": 0, "state": None, "valid": None, "gate": None}
    resume_path = os.path.join(out_dir, f"resume_{stem}.pt")
    resumed_at, start_epoch, sec_before = [], 1, 0.0
    r = load_resume(resume_path, args)
    if r is not None:
        model.load_state_dict(r["model"])
        optimizer.load_state_dict(r["optimizer"])
        set_rng_state(gen, r["rng"])
        history, best, sec_before = r["history"], r["best"], r["elapsed_sec"]
        resumed_at = r["resumed_at"] + [{"epoch": r["epoch"], "code_commit": code_commit(),
                                         "previous_code_commit": r["code_commit"]}]
        start_epoch = r["epoch"] + 1
        print(f"resumed from {resume_path}: epoch {r['epoch']} done, best valid MRR {best['mrr']:.4f} "
              f"(epoch {best['epoch']})" + ("" if r["code_commit"] == code_commit() else
                                            f" | WARNING: written by code {r['code_commit']}, now {code_commit()}"),
              flush=True)
    train_dev = train.to(device)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(train), generator=gen).to(device)
        sums, n_batches = {}, 0
        for i in range(0, len(perm), args.batch_size):
            b = train_dev[perm[i:i + args.batch_size]]
            loss, stats = in_batch_loss(model, feats, b[:, 0], b[:, 1], b[:, 2], train_index,
                                        args.margin, args.adv_temp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            stats["loss"] = loss.item()
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1
        row = {"epoch": epoch, **{k: v / n_batches for k, v in sums.items()}, "sec": time.time() - t0}
        print(f"epoch {epoch:4d}/{args.epochs} | loss {row['loss']:.4f} (tail {row['loss_tail']:.4f}, "
              f"head {row['loss_head']:.4f}) | eff. negatives tail {row['eff_neg_tail']:.1f} / "
              f"{row['n_neg_tail']:.0f}, head {row['eff_neg_head']:.1f} / {row['n_neg_head']:.0f} | "
              f"{row['sec']:.1f}s", flush=True)

        if epoch % args.valid_every == 0 or epoch == args.epochs:
            m = strip_ranks(evaluate(valid, f"valid@{epoch}"))
            row["valid"] = m
            row["gate"] = gate_stats(model, feats, n_rel)
            row["depth_stats_norm"] = depth_stats_norm(model, feats)
            g = row["gate"]
            gate_txt = "" if g is None else (
                f" | depth weight {g['mean']:.3f}"
                + (f" (sd entities {g['std_entities']:.3f}" if "std_entities" in g else "")
                + (f", sd relations {g['std_relations']:.3f}" if "std_relations" in g else "")
                + (")" if "std_entities" in g else "")) + (
                "" if row["depth_stats_norm"] is None else f" | |W_s s| {row['depth_stats_norm']:.3f}")
            print(f"valid @ epoch {epoch}: {format_metrics(m)}{gate_txt}", flush=True)
            if m["mrr"] > best["mrr"]:
                best = {"mrr": m["mrr"], "epoch": epoch, "valid": m, "gate": g,
                        "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}
                print(f"  new best valid MRR {m['mrr']:.4f}", flush=True)
        history.append(row)
        if not args.no_resume and "valid" in row and epoch < args.epochs:
            save_resume(resume_path, {
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": rng_state(gen), "history": history, "best": best, "resume_key": resume_key(args),
                "code_commit": code_commit(), "resumed_at": resumed_at,
                "elapsed_sec": sec_before + time.time() - t_start})
            print(f"resume state saved: {resume_path} (epoch {epoch})", flush=True)

    metrics = {
        "model": "GeometryMMKG", "dataset": dataset, "variant": args.variant, "model_kwargs": model_kwargs,
        "options": options, "options_tag": tag, "dev": args.dev, "depth_stats": depth_stats_info,
        "tune": args.tune, "config": {"lr": args.lr, "dropout": args.dropout} if args.tune else None,
        "epoch_budget": {"epochs": args.epochs, "valid_every": args.valid_every,
                         "dataset_default": [budget_epochs, budget_valid]},
        "seed": args.seed, "args": vars(args), "code_commit": code_commit(), "n_parameters": n_params,
        "n_entities": n_ent, "n_relations": n_rel, "n_missing_image": n_missing,
        "n_train": len(train), "n_valid_used": len(valid), "valid_subset_seed": VALID_SUBSET_SEED if valid_max else None,
        "best_epoch": best["epoch"], "best_valid": best["valid"], "best_gate": best["gate"],
        "resumed_at": resumed_at,
    }
    if args.dev:
        print(f"[dev] best epoch {best['epoch']}, valid MRR {best['mrr']:.4f}; test not evaluated.", flush=True)
    else:
        model.load_state_dict(best["state"])
        print(f"Best epoch {best['epoch']} (valid MRR {best['mrr']:.4f}); evaluating on test once.", flush=True)
        test_m = evaluate(test, "test", return_ranks=True)
        print(f"test: {format_metrics(test_m)}", flush=True)
        save_ranks(test_m["ranks"], os.path.join(out_dir, f"ranks_{stem}_test.csv"))
        metrics["test"] = strip_ranks(test_m)
        if not args.no_ckpt:
            torch.save({"state_dict": best["state"], "variant": args.variant, "model_kwargs": model_kwargs,
                        "options": options, "args": vars(args), "epoch": best["epoch"]},
                       os.path.join(out_dir, f"ckpt_{stem}.pt"))
    metrics["history"] = history
    metrics["total_sec"] = sec_before + time.time() - t_start
    with open(os.path.join(out_dir, f"metrics_{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if os.path.exists(resume_path):
        os.remove(resume_path)   # finished: the metrics (and the best checkpoint) are the result
    print(f"Wrote metrics{'' if args.dev else ' / ranks' + ('' if args.no_ckpt else ' / checkpoint')} to "
          f"{out_dir} ({(time.time() - t_start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
