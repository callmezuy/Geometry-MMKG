"""
Train and evaluate GeometryMMKG on one v0 dataset (RESEARCH_PLAN.md P2.1, P2.8, section 4).

  python scripts/train.py --data data/Spatial-VRD --variant full --epochs 100 --seed 42
  python scripts/train.py --data data/Spatial-VRD --variant full --rel-gate --dev     # development run

* Variants (section 3.1): full, no_depth, concat, mean, fixed, constant (core/models/end_to_end.py VARIANTS).
* Development options (P2.8, off by default): --rel-gate, --input-norm, --depth-dropout P, --text-proj.
  The same options can be passed to every variant; each variant uses those that apply to it
  (`effective_options`), and the options tag (folder / file names) is the same for all of them.
* Training: in-batch negatives on both sides, train-only false-negative filter, self-adversarial weighted
  margin ranking (core/models/kg_reasoning.py).
* Model selection: filtered MRR (head + tail) on valid every --valid-every epochs and after the last epoch;
  the best checkpoint is evaluated on test once. Spatial-VG validates on a fixed 5,000-triple subset (4.9),
  the same subset for every variant and seed. Each validation also logs the gate's weight on depth.
* --dev: model development (P2.8). Valid only: test is never evaluated, no ranks, no checkpoint.
* Output: metrics_<stem>.json, plus ranks_<stem>_test.csv (for scripts/compute_metrics.py) and ckpt_<stem>.pt
  outside --dev; stem = geometry_<dataset>_<variant>[_<options tag>]_seed<seed>.
  Default folder: results/geometry/<dataset>/<variant>/, or results/geometry-dev/<dataset>/<tag>/<variant>/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from core.data_engineering.fast_dataset import (  # noqa: E402
    TripleBatches, get_all_true_triplets, load_entity_features, load_triples, read_count)
from core.models.end_to_end import VARIANTS, GeometryMMKGModel, effective_options  # noqa: E402
from core.models.evaluation import evaluate_link_prediction, format_metrics, save_ranks  # noqa: E402
from core.models.kg_reasoning import TripleIndex, in_batch_loss  # noqa: E402
from core.reproducibility import set_seed  # noqa: E402

VALID_SUBSET_SEED = 0       # fixed: every variant / seed validates on the same subset
VG_VALID_MAX = 5000         # RESEARCH_PLAN.md 4.9
GATE_SAMPLE = 4096          # entities (with an image) used for the gate statistics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset folder, e.g. data/Spatial-VRD")
    p.add_argument("--variant", default="full", choices=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--margin", type=float, default=5.0)
    p.add_argument("--adv-temp", type=float, default=1.0, help="self-adversarial temperature")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rel-gate", action="store_true", help="per-relation gate bias (P2.2)")
    p.add_argument("--input-norm", action="store_true", help="separate LayerNorm on RGB and depth before fusion")
    p.add_argument("--depth-dropout", type=float, default=0.0, help="drop depth per entity while training")
    p.add_argument("--text-proj", action="store_true", help="learned projection of text (identity at init)")
    p.add_argument("--dev", action="store_true", help="development run: valid only, never test")
    p.add_argument("--valid-every", type=int, default=10, help="epochs between validations")
    p.add_argument("--valid-max", type=int, default=None,
                   help=f"validate on a fixed random subset of this size (0 = all; default {VG_VALID_MAX} for "
                        "Spatial-VG, else all)")
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--keep-on-gpu", choices=["auto", "yes", "no"], default="auto",
                   help="feature / fused tables on the GPU (auto: if < 50%% of free memory)")
    p.add_argument("--out", default=None, help="output folder (default: see above)")
    p.add_argument("--no-ckpt", action="store_true", help="do not write the best checkpoint")
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
    return "-".join(parts) or "base"


def code_commit():
    try:
        return subprocess.run(["git", "-C", CODE_DIR, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


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
    tag = options_tag(args)
    stem = f"geometry_{dataset}_{args.variant}{'' if tag == 'base' else '_' + tag}_seed{args.seed}"
    if args.out:
        out_dir = args.out
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
    print(f"{n_ent:,} entities ({n_missing:,} without image), {n_rel} relations | train {len(train):,} | "
          f"valid {len(valid):,}{' (subset)' if valid_max else ''}"
          f"{'' if test is None else f' | test {len(test):,}'} | features stored on {feats.store}", flush=True)

    model_kwargs = dict(VARIANTS[args.variant])
    options = effective_options(model_kwargs, args.rel_gate, args.input_norm, args.depth_dropout, args.text_proj)
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
            device, keep_on_gpu=keep, return_ranks=return_ranks, desc=desc, all_mask=feats.mask)

    history, best = [], {"mrr": -1.0, "epoch": 0, "state": None, "valid": None, "gate": None}
    train_dev = train.to(device)
    for epoch in range(1, args.epochs + 1):
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
            g = row["gate"]
            gate_txt = "" if g is None else (
                f" | depth weight {g['mean']:.3f}"
                + (f" (sd entities {g['std_entities']:.3f}" if "std_entities" in g else "")
                + (f", sd relations {g['std_relations']:.3f}" if "std_relations" in g else "")
                + (")" if "std_entities" in g else ""))
            print(f"valid @ epoch {epoch}: {format_metrics(m)}{gate_txt}", flush=True)
            if m["mrr"] > best["mrr"]:
                best = {"mrr": m["mrr"], "epoch": epoch, "valid": m, "gate": g,
                        "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}
                print(f"  new best valid MRR {m['mrr']:.4f}", flush=True)
        history.append(row)

    metrics = {
        "model": "GeometryMMKG", "dataset": dataset, "variant": args.variant, "model_kwargs": model_kwargs,
        "options": options, "options_tag": tag, "dev": args.dev,
        "seed": args.seed, "args": vars(args), "code_commit": code_commit(), "n_parameters": n_params,
        "n_entities": n_ent, "n_relations": n_rel, "n_missing_image": n_missing,
        "n_train": len(train), "n_valid_used": len(valid), "valid_subset_seed": VALID_SUBSET_SEED if valid_max else None,
        "best_epoch": best["epoch"], "best_valid": best["valid"], "best_gate": best["gate"],
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
    metrics["total_sec"] = time.time() - t_start
    with open(os.path.join(out_dir, f"metrics_{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics{'' if args.dev else ' / ranks' + ('' if args.no_ckpt else ' / checkpoint')} to "
          f"{out_dir} ({(time.time() - t_start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
