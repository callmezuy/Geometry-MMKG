"""Exact-resume check of the CMR patch (baselines/patches/CMR.diff, RESEARCH_PLAN.md P3.7). Usage, from the CMR
checkout with data/tiny prepared (scripts/cmr_molab_run.sh does both): python cmr_resume_check.py ROOT
Run A (3 epochs straight) vs run B (killed once resume.pt holds an epoch, then --resume). The valid
metric line of every epoch must be identical. Run from the CMR checkout, CPU. On Windows resume.pt is never opened
while B runs (os.replace onto an open file fails there): B is watched through its log and the file's mtime."""
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import torch

ROOT = sys.argv[1]
BASE = [sys.executable, "main.py", "--task", "FB15k237_ind", "--train-path", "./data/tiny/train.txt.json",
        "--valid-path", "./data/tiny/valid.txt.json", "--pretrained-model", "./PLMs/bert-base-uncased",
        "--pooling", "mean", "--lr", "1e-5", "--use-link-graph", "--batch-size", "8", "--pre-batch", "2",
        "--epochs", "3", "--workers", "2", "--mm", "--prefix", "4", "--freeze-vit", "--finetune-t",
        "--print-freq", "4", "--early-stopping", "10", "--max-to-keep", "1"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUNBUFFERED="1")


def valid_lines(run_dir):
    out = {}
    for line in open(os.path.join(run_dir, "log.txt"), encoding="utf-8"):
        if "valid metric" in line:
            out[int(re.search(r"Epoch (\d+)", line).group(1))] = re.sub(r"^\[[^\]]*\] ", "", line.strip())
    return out


assert not os.path.exists(ROOT), ROOT  # a fresh folder: nothing is deleted
r = subprocess.run(BASE + ["--model-dir", f"{ROOT}/A/"], env=env, capture_output=True, text=True)
if r.returncode:
    raise SystemExit("run A failed:\n" + r.stderr[-3000:])
a = valid_lines(glob.glob(f"{ROOT}/A/*/")[0])

p = subprocess.Popen(BASE + ["--model-dir", f"{ROOT}/B/"], env=env, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
t_seen = None
while True:
    time.sleep(0.5)
    if p.poll() is not None:
        raise SystemExit("run B ended before it could be interrupted")
    d = glob.glob(f"{ROOT}/B/*/")
    if not d or not os.path.exists(os.path.join(d[0], "log.txt")):
        continue
    res = os.path.join(d[0], "resume.pt")
    if t_seen is None and len(valid_lines(d[0])) >= 1:
        t_seen = time.time()
    if t_seen and os.path.exists(res) and os.path.getmtime(res) > t_seen - 1 and \
            time.time() - os.path.getmtime(res) > 2 and not os.path.exists(res + ".tmp"):
        break
p.kill()
p.wait()
b_dir = d[0]
print("killed B after epoch", torch.load(res, map_location="cpu", weights_only=False)["epoch"])
r = subprocess.run(BASE + ["--model-dir", f"{ROOT}/ignored/", "--resume", b_dir], env=env, capture_output=True,
                   text=True)
if r.returncode:
    raise SystemExit("resumed run failed:\n" + r.stderr[-3000:])
b = valid_lines(b_dir)
print("A:", *[a[k] for k in sorted(a)], sep="\n  ")
print("B:", *[b[k] for k in sorted(b)], sep="\n  ")
same = sorted(a) == [0, 1, 2] and all(a[k] == b.get(k) for k in a)
print("identical per epoch:", same)
sys.exit(0 if same else 1)
