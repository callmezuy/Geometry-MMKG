"""
Measure per-epoch training time and validation time of the remaining baselines on Molab (RESEARCH_PLAN.md P3.6,
P3.7, P4 budget). Nothing here changes baseline code: the KGE baselines run their own training scripts for a few
epochs; CMR and MKGformer (raw-image models) are split into a GPU step benchmark on synthetic pixels and a CPU
image-pipeline benchmark on real images, because their epoch time is the slower of the two.

Run from the repository root (Code/ locally, /marimo/Geometry-MMKG on Molab). Each call appends one JSON line to
results/timing/baseline_timing.jsonl. Every subcommand finishes in minutes; start it from a Molab cell, e.g.
    subprocess.run([sys.executable, "scripts/time_baselines.py", "momok", "--dataset", "FB15k-237-IMG"], check=True)

Subcommands:
  momok   --dataset D [--epochs 4]     clone zjukg/MoMoK (pinned) into --work, write our v0 data in its format, run
                                       its train.py for a few epochs (no evaluation), then time one validation pass
                                       with its own ConvECorpus.get_validation_pred in a separate process.
  snag    --dataset D [--epochs 4]     baselines/SNAG/SNAG_MKGC/run_mmkgc.py, author flags (run_base.sh), control
                                       table, one valid evaluation at the last epoch (ValidSelector).
  native  --dataset D [--epochs 4]     baselines/NativE/run_adv_wgan_gp.py, author flags (run_mkgy.sh), control table.
  cmr-gpu [--steps 30]                 CMR CustomMMModel (2 x BERT-base + ViT-B/16, prefix 4), batch 768, AMP,
                                       synthetic tokens/pixels, all parameters trainable as in the released code.
  mkgformer-gpu --dataset D [--steps 30]  MKGformer UnimoKGC (BERT-base + CLIP ViT-B/32), batch 96, seq 64, fp32,
                                       vocabulary 30522 + entities + relations. Needs a transformers-5 import shim
                                       (count/timing only; real runs need the pinned old environment).
  decode  --images ZIP_OR_DIR [--n 2000] [--workers 4 16]
                                       CPU throughput of the two image pipelines on real images:
                                       cmr = PIL open, RGB, resize 384, ViT processor (2 images per example);
                                       mkgformer = 1 CLIP 224 + 3 aux 128 + 3 rcnn 64 per example (7 images).

Epoch time per baseline = max(GPU time, CPU image time / workers) for CMR and MKGformer; the report says which.
"""
import argparse
import glob
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "timing", "baseline_timing.jsonl")
HF_REPOS = {
    "FB15k-237-IMG": "callmezuy/fb15k237-img",
    "FB15K-237-ind": "callmezuy/fb15k237-ind",
    "MKG-Y": "callmezuy/mkg-y",
    "Spatial-VRD": "callmezuy/spatial-vrd",
    "Spatial-VRD-Trans": "callmezuy/spatial-vrd-trans",
}
MOMOK_URL, MOMOK_COMMIT = "https://github.com/zjukg/MoMoK.git", "99c2df114d48c79708ea0608644b685183d81bd9"
CMR_URL, CMR_COMMIT = "https://github.com/OreOZhao/CMR.git", "bcd23a66fb5787f5968b7615a836c8b48361b5ec"
OPENKE = ["entity2id.txt", "relation2id.txt", "train2id.txt", "valid2id.txt", "test2id.txt"]


def record(entry):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        import torch
        if torch.cuda.is_available():
            entry["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=1))


def fetch_data(ds, work):
    """Download the v0 files a feature-table baseline needs (public repos, no token)."""
    from huggingface_hub import snapshot_download
    d = os.path.join(work, "data", ds)
    snapshot_download(repo_id=HF_REPOS[ds], repo_type="dataset", revision="v0", local_dir=d,
                      allow_patterns=OPENKE + ["ent_rgb_emb.pt", "ent_text_emb.pt", "ent_img_mask.pt",
                                               "ent_fill.pt", "entity2textlong.txt", "entity2text.txt"])
    return d


def read_openke(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [line.split() for line in f if line.strip()]
    assert len(rows) == n, (path, n, len(rows))
    return rows


def clone(url, commit, dst):
    if not os.path.isdir(os.path.join(dst, ".git")):
        shutil.rmtree(dst, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", url, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", commit], check=True)


def build_openke_lib(mmkgc_dir):
    """Same build step as notebooks/native.py and snag-mkgy-sanity.py."""
    if glob.glob(os.path.join(mmkgc_dir, "release", "Base.so")):
        return
    sh = os.path.join(mmkgc_dir, "make.sh")
    with open(sh, encoding="utf-8") as f:
        txt = f.read().replace("\r\n", "\n").replace("mkdir release", "mkdir -p release")
    with open(sh, "w", encoding="utf-8") as f:
        f.write(txt)
    subprocess.run(["bash", "make.sh"], cwd=mmkgc_dir, check=True)


TQDM_COUNT = re.compile(r"(\d+)/(\d+) \[")


def run_timed(cmd, cwd, epochs, env=None, marker=None):
    """Run a training script, timestamp tqdm epoch counters (k/N) and optional marker lines.

    The tqdm counter k appears when epoch k-1 (0-based) has finished, including any validation done inside
    that loop iteration.
    """
    env = dict(os.environ if env is None else env, PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    first_seen, markers, tail, buf = {}, [], [], b""
    while True:
        ch = p.stdout.read(1)
        if not ch:
            break
        if ch in (b"\n", b"\r"):
            line = buf.decode("utf-8", "replace")
            buf = b""
            now = time.time() - t0
            for k, n in TQDM_COUNT.findall(line):
                if int(n) == epochs:  # the training bar only (test/eval bars have other totals)
                    first_seen.setdefault(int(k), now)
            if marker and marker in line:
                markers.append((now, line.strip()[:200]))
            if line.strip():
                tail = (tail + [line.strip()[:300]])[-15:]
        else:
            buf += ch
    rc = p.wait()
    total = time.time() - t0
    if rc != 0:
        print("\n".join(tail))
        raise SystemExit(f"command failed with exit code {rc}: {' '.join(cmd)}")
    return first_seen, markers, total, tail


def epoch_stats(first_seen, epochs):
    """Median seconds per epoch from epochs 2..E-1 (skips warm-up and the last epoch, which holds validation)."""
    ts = [first_seen.get(k) for k in range(0, epochs + 1)]
    diffs = [ts[k] - ts[k - 1] for k in range(2, epochs) if ts[k] is not None and ts[k - 1] is not None]
    last = ts[epochs] - ts[epochs - 1] if ts[epochs] is not None and ts[epochs - 1] is not None else None
    return (statistics.median(diffs) if diffs else None), diffs, last


# ------------------------------------------------------------------------------------------------- MoMoK
def momok(a):
    work = os.path.abspath(a.work)
    src = os.path.join(work, "MoMoK")
    clone(MOMOK_URL, MOMOK_COMMIT, src)
    d = fetch_data(a.dataset, work)
    # MoMoK picks its feature reshape by dataset-name substring (RelMoE.__init__). For 768-d image + 768-d text
    # the matching author branch is "FB15K-237" (12x64 -> AdaptiveAvgPool (4,64) -> 256). The name must not contain
    # "MKG" (checked first; it expects 384-d text). Timing only: the cost is the same for every 768/768 dataset.
    name = "FB15K-237-timing-" + a.dataset.replace("MKG", "mkg")
    dst = os.path.join(src, "datasets", name)
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(src, "checkpoint", name), exist_ok=True)
    ents = read_openke(os.path.join(d, "entity2id.txt"))
    rels = read_openke(os.path.join(d, "relation2id.txt"))
    with open(os.path.join(dst, "entity2id.txt"), "w") as f:
        f.writelines(f"e{i} {i}\n" for _, i in ents)
    with open(os.path.join(dst, "relation2id.txt"), "w") as f:
        f.writelines(f"r{i} {i}\n" for _, i in rels)
    for split in ("train", "valid", "test"):
        rows = read_openke(os.path.join(d, f"{split}2id.txt"))  # OpenKE order: h t r
        with open(os.path.join(dst, f"{split}.txt"), "w") as f:
            f.writelines(f"e{h} r{r} e{t}\n" for h, t, r in rows)
    import torch
    for ours, theirs in (("ent_rgb_emb.pt", "img_features.pth"), ("ent_text_emb.pt", "text_features.pth")):
        x = torch.load(os.path.join(d, ours), map_location="cpu", weights_only=True)
        assert x.shape == (len(ents), 768), (ours, tuple(x.shape))
        torch.save(x, os.path.join(dst, theirs))

    # 1) the authors' train.py, MKG-Y config (README: lr 5e-4, mu 1e-4, dim 300), no evaluation inside the run
    cmd = [sys.executable, "train.py", "--cuda", "0", "--lr", "0.0005", "--mu", "0.0001", "--dim", "300",
           "--dataset", name, "--epochs", str(a.epochs), "--eval_freq", "1000000", "--save", "0"]
    first_seen, _, total, _ = run_timed(cmd, src, a.epochs)
    med, diffs, last = epoch_stats(first_seen, a.epochs)
    # 2) one validation pass with the authors' evaluation code (separate process: train.py runs at import time)
    drv = (
        "import sys, time, json\n"
        "from argparse import Namespace\n"
        "import torch, torch.nn.functional as F\n"
        "from utils.data_util import load_data\n"
        "from utils.data_loader import ConvECorpus\n"
        "from models.RelMoE import RelMoE\n"
        f"name = {name!r}\n"
        "e2i, r2i, img, txt, tr, va, te = load_data(name)\n"
        "args = Namespace(batch_size=1024, device='cuda:0', dataset=name, dim=300, r_dim=256, n_exp=3, pre_trained=0,\n"
        "                 img=F.normalize(torch.Tensor(img), p=2, dim=1), desp=F.normalize(torch.Tensor(txt), p=2, dim=1),\n"
        "                 entity2id=e2i, relation2id=r2i)\n"
        "corpus = ConvECorpus(args, tr, va, te, e2i, r2i)\n"
        "model = RelMoE(args).to('cuda:0').eval()\n"
        "out = {'n_params': sum(p.numel() for p in model.parameters()),\n"
        "       'img_rows': model.img_entity_embeddings.weight.shape[0], 'n_entities': len(e2i)}\n"
        "torch.cuda.synchronize(); t = time.time()\n"
        "try:\n"
        "    with torch.no_grad():\n"
        "        res = corpus.get_validation_pred(model, 'val')\n"
        "    out['valid_eval_s'] = time.time() - t\n"
        "    out['returns'] = type(res).__name__\n"
        "except Exception as e:\n"
        "    out['valid_eval_error'] = repr(e)[:300]\n"
        "out['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30\n"
        "print('MOMOK_EVAL ' + json.dumps(out))\n"
    )
    res = subprocess.run([sys.executable, "-c", drv], cwd=src, capture_output=True, text=True)
    ev = {}
    for line in res.stdout.splitlines():
        if line.startswith("MOMOK_EVAL "):
            ev = json.loads(line[len("MOMOK_EVAL "):])
    if not ev:
        ev = {"driver_error": (res.stderr or res.stdout)[-800:]}
    record({"baseline": "MoMoK", "dataset": a.dataset, "epochs_run": a.epochs, "s_per_epoch": med,
            "epoch_diffs": diffs, "wall_s": total,
            "projected_h_2000_epochs": med * 2000 / 3600 if med else None, **ev,
            "config": "dim 300, lr 5e-4, mu 1e-4, batch 1024 (README MKG-Y), branch FB15K-237"})


# ------------------------------------------------------------------------------------------- SNAG / NativE
def snag(a):
    work = os.path.abspath(a.work)
    d = fetch_data(a.dataset, work)
    snag_dir = os.path.join(ROOT, "baselines", "SNAG", "SNAG_MKGC")
    build_openke_lib(os.path.join(snag_dir, "mmkgc"))
    try:
        import unidecode  # noqa: F401  (SNAG's torchlight logger)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unidecode"], check=True)
    name = f"{a.dataset}-v0-timing"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prepare_snag_data.py"), "--data", d,
                    "--snag", snag_dir, "--name", name, "--visual", "rgb"], check=True)
    pool = "1" if "MKG" in name else "0"  # MKG names never pool (AdvMixRotatE); others need -use_pool 0
    for p in glob.glob(os.path.join(snag_dir, "checkpoint", "*timing*")):
        os.remove(p)
    cmd = [sys.executable, "run_mmkgc.py", f"-dataset={name}", "-num_proj=1", "-use_intermediate=0",
           "-joint_way=Mformer_hd_mean", "-batch_size=1024", "-margin=12", f"-epoch={a.epochs}", "-dim=128",
           f"-save={name}-timing", "-neg_num=32", "-noise_update=epoch", "-noise_ratio=0.2", "-mask_ratio=0.7",
           f"-use_pool={pool}", "-add_noise=1", "-learning_rate=1e-4", "-num_hidden_layers=1",
           "-num_attention_heads=2", f"-exp_id=timing_{a.dataset}", "-select_on=valid",
           f"-valid_steps={a.epochs}", "-patience=0", "-seed=42"]
    first_seen, markers, total, _ = run_timed(cmd, snag_dir, a.epochs, marker="[ValidSelector] epoch")
    med, diffs, last = epoch_stats(first_seen, a.epochs)
    record({"baseline": "SNAG", "dataset": a.dataset, "epochs_run": a.epochs, "s_per_epoch": med,
            "epoch_diffs": diffs, "last_epoch_with_valid_s": last,
            "valid_eval_s": (last - med) if (last and med) else None, "wall_s_incl_test": total,
            "projected_h_per_1000_epochs": med * 1000 / 3600 if med else None,
            "valid_evals_per_1000_epochs_at_steps_100": 10,
            "config": "run_base.sh flags (dim 128, batch 1024, neg 32, margin 12, lr 1e-4), control table"})


def native(a):
    work = os.path.abspath(a.work)
    d = fetch_data(a.dataset, work)
    nat = os.path.join(ROOT, "baselines", "NativE")
    build_openke_lib(os.path.join(nat, "mmkgc"))
    name = f"{a.dataset}__timing"
    bench = os.path.join(nat, "benchmarks", name)
    os.makedirs(bench, exist_ok=True)
    for fn in OPENKE:
        shutil.copyfile(os.path.join(d, fn), os.path.join(bench, fn))
    os.makedirs(os.path.join(nat, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(nat, "checkpoint"), exist_ok=True)
    for ours, suf in (("ent_rgb_emb.pt", "visual"), ("ent_text_emb.pt", "textual")):
        dst = os.path.join(nat, "embeddings", f"{name}-{suf}.pth")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.join(d, ours), dst)
    cmd = [sys.executable, "run_adv_wgan_gp.py", f"-dataset={name}", "-batch_size=1024", "-margin=4",
           f"-epoch={a.epochs}", "-dim=250", f"-save={os.path.join(nat, 'checkpoint', name + '.ckpt')}",
           "-neg_num=128", "-mu=0.0001", "-learning_rate=0.0001", "-lrg=0.0001", "-seed=42", "-select_on=valid",
           f"-valid_steps={a.epochs}"]
    first_seen, markers, total, _ = run_timed(cmd, nat, a.epochs, marker="[ValidSelector] epoch")
    med, diffs, last = epoch_stats(first_seen, a.epochs)
    record({"baseline": "NativE", "dataset": a.dataset, "epochs_run": a.epochs, "s_per_epoch": med,
            "epoch_diffs": diffs, "last_epoch_with_valid_s": last,
            "valid_eval_s": (last - med) if (last and med) else None, "wall_s_incl_test": total,
            "projected_h_per_1000_epochs": med * 1000 / 3600 if med else None,
            "config": "run_mkgy.sh flags (dim 250, batch 1024, neg 128, margin 4, lr 1e-4), control table"})


# ---------------------------------------------------------------------------------------------- CMR GPU
def cmr_gpu(a):
    import types
    import torch
    from huggingface_hub import snapshot_download
    work = os.path.abspath(a.work)
    src = os.path.join(work, "CMR")
    clone(CMR_URL, CMR_COMMIT, src)
    plms = os.path.join(work, "cmr_run", "PLMs")
    os.makedirs(plms, exist_ok=True)
    for repo, local in (("bert-base-uncased", "bert-base-uncased"), ("google/vit-base-patch16-224",
                                                                     "vit-base-patch16-224")):
        snapshot_download(repo_id=repo, local_dir=os.path.join(plms, local),
                          allow_patterns=["*.json", "*.txt", "*.safetensors"])
    os.chdir(os.path.dirname(plms))  # CMR loads './PLMs/vit-base-patch16-224' relative to the working dir
    # models.py imports triplet_mask -> config (argparse) and the entity dict; neither is used by forward().
    sys.modules["triplet_mask"] = types.SimpleNamespace(construct_mask=None)
    sys.path.insert(0, src)
    from models import CustomMMModel
    B, L, prefix = a.batch or 768, 50, 4
    args = types.SimpleNamespace(pretrained_model="./PLMs/bert-base-uncased", t=0.05, finetune_t=True,
                                 additive_margin=0.0, batch_size=B, pre_batch=0, prefix=prefix, pooling="mean",
                                 use_self_negative=False, pre_batch_weight=0.5)
    model = CustomMMModel(args).cuda().train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ce = torch.nn.CrossEntropyLoss()
    ids = lambda: torch.randint(1000, 30000, (B, L), device="cuda")
    mask = torch.ones(B, L, dtype=torch.long, device="cuda")
    pix = lambda: torch.randn(B, 3, 224, 224, device="cuda")

    def step():
        batch = dict(hr_token_ids=ids(), hr_mask=mask, tr_token_ids=ids(), tr_mask=mask, head_pixel_values=pix(),
                     tail_pixel_values=pix(), head_token_ids=ids(), head_mask=mask, tail_token_ids=ids(),
                     tail_mask=mask, batch_data=None)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        lg = model.compute_logits(out, {"batch_data": None})
        lab = lg["labels"]
        loss = ce(lg["logits"], lab) + ce(lg["logits"][:, :B].t(), lab) + ce(lg["logits_mm"], lab) \
            + ce(lg["logits_mm"][:, :B].t(), lab)  # trainer.py with args.mem False (both directions)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    s_step = (time.time() - t) / a.steps
    steps = {"FB15K-237-ind": 215082 * 2 // B, "Spatial-VRD": 76959 * 2 // B}
    record({"baseline": "CMR", "part": "gpu", "batch": B, "s_per_step": s_step, "n_params": n_params,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_min_per_epoch": {k: v * s_step / 60 for k, v in steps.items()},
            "note": "synthetic tokens (len 50) and pixels; pre-batch negatives off (tiny cost); ViT trainable as "
                    "in the released code (paper says frozen)"})


# ----------------------------------------------------------------------------------------- MKGformer GPU
def mkgformer_gpu(a):
    import torch
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    from transformers import BertConfig, CLIPConfig
    mu.apply_chunking_to_forward = pu.apply_chunking_to_forward  # timing-only shim for transformers 5
    sys.path.insert(0, os.path.join(ROOT, "baselines", "MKGformer", "MKG"))
    from models.model import UnimoKGC
    vcfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32").vision_config
    tcfg = BertConfig.from_pretrained("bert-base-uncased")
    vcfg.device = "cuda"
    vcfg.torchscript = tcfg.torchscript = False
    n_ent, n_rel = {"FB15k-237-IMG": (14541, 237), "Spatial-VRD-Trans": (93809, 52)}[a.dataset]
    model = UnimoKGC(vcfg, tcfg)
    vocab = 30522 + n_ent + n_rel
    model.resize_token_embeddings(vocab)
    model = model.cuda().train()
    n_params = sum(p.numel() for p in model.parameters())
    B, L = a.batch or 96, 64
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5)
    st = 30522 + n_rel  # entity token slice; exact offset does not change the cost

    def step():
        ids = torch.randint(1000, 30000, (B, L), device="cuda")
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), token_type_ids=torch.zeros_like(ids),
                    pixel_values=torch.randn(B, 3, 224, 224, device="cuda"),
                    aux_values=torch.randn(B, 3, 3, 128, 128, device="cuda"),
                    rcnn_values=torch.randn(B, 3, 3, 64, 64, device="cuda"), return_dict=True)
        logits = out.logits[torch.arange(B), L // 2][:, st:st + n_ent]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    s_step = (time.time() - t) / a.steps
    n_train = {"FB15k-237-IMG": 272115, "Spatial-VRD-Trans": 77083}[a.dataset] * 2  # head + tail examples
    record({"baseline": "MKGformer", "part": "gpu", "dataset": a.dataset, "batch": B, "s_per_step": s_step,
            "n_params": n_params, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_min_per_epoch": n_train / B * s_step / 60,
            "note": "fp32 (script has no --precision), synthetic inputs, 7 image slots per example"})


# ------------------------------------------------------------------------------------------ CPU decode
def _decode_worker(job):
    """Run one image pipeline on a list of encoded images; returns images/s for this worker."""
    kind, blobs = job
    from PIL import Image
    from transformers import AutoImageProcessor, CLIPImageProcessor
    if kind == "cmr":
        proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        t = time.time()
        for b in blobs:
            im = Image.open(io.BytesIO(b)).convert("RGB").resize((384, 384))
            proc(images=im, return_tensors="pt")
    else:
        full = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        aux = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", size={"shortest_edge": 128},
                                                 crop_size={"height": 128, "width": 128})
        rcnn = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", size={"shortest_edge": 64},
                                                  crop_size={"height": 64, "width": 64})
        t = time.time()
        for i, b in enumerate(blobs):
            im = Image.open(io.BytesIO(b)).convert("RGB")
            (full if i % 7 == 0 else aux if i % 7 < 4 else rcnn)(images=im, return_tensors="pt")
    return len(blobs) / (time.time() - t)


def decode(a):
    import random
    from multiprocessing import Pool
    exts = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
    blobs = []
    if os.path.isdir(a.images):
        files = [p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True) if p.lower().endswith(exts)]
        random.Random(0).shuffle(files)
        for p in files[: a.n]:
            with open(p, "rb") as f:
                blobs.append(f.read())
    else:
        with zipfile.ZipFile(a.images) as z:
            names = [n for n in z.namelist() if n.lower().endswith(exts)]
            random.Random(0).shuffle(names)
            blobs = [z.read(n) for n in names[: a.n]]
    mean_kb = sum(map(len, blobs)) / len(blobs) / 1024
    res = {}
    for kind in ("cmr", "mkgformer"):
        for w in a.workers:
            chunk = [blobs[i::w] for i in range(w)]
            with Pool(w) as pool:
                rates = pool.map(_decode_worker, [(kind, c) for c in chunk])
            res[f"{kind}_img_per_s_workers{w}"] = sum(rates)
    per_epoch = {}
    for w in a.workers:
        r_cmr, r_mkg = res[f"cmr_img_per_s_workers{w}"], res[f"mkgformer_img_per_s_workers{w}"]
        per_epoch[f"workers{w}"] = {
            "CMR FB15K-237-ind (2 img x 430,164 ex)": 2 * 430164 / r_cmr / 60,
            "MKGformer FB15k-237-IMG (<=7 img x 544,230 ex)": 7 * 544230 / r_mkg / 60,
        }
    record({"baseline": "CMR+MKGformer", "part": "cpu-decode", "source": os.path.basename(a.images),
            "n_images": len(blobs), "mean_kb": mean_kb, "cpus": os.cpu_count(), **res,
            "cpu_min_per_epoch": per_epoch,
            "note": "MKGformer: most FB15k entities have >=7 images (median 10), so 7 decodes per example"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["momok", "snag", "native", "cmr-gpu", "mkgformer-gpu", "decode"])
    ap.add_argument("--dataset", choices=sorted(HF_REPOS), default="FB15k-237-IMG")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--work", default=os.path.join(ROOT, "..", "timing"))
    ap.add_argument("--images", help="zip or folder of real images (decode)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 16])
    a = ap.parse_args()
    if a.what in ("momok", "snag", "native") and a.epochs < 4:
        ap.error("--epochs must be >= 4 (epochs 2..E-1 give the median; the last one includes validation)")
    {"momok": momok, "snag": snag, "native": native, "cmr-gpu": cmr_gpu, "mkgformer-gpu": mkgformer_gpu,
     "decode": decode}[a.what](a)


if __name__ == "__main__":
    main()
