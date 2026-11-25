"""Exact-resume check of the MoMoK patch (baselines/patches/MoMoK.diff, RESEARCH_PLAN.md P3.6). Run from a MoMoK
checkout (scripts/setup_baseline.py momok) where scripts/prepare_momok_data.py wrote MKG-Y (default name). 25/11/2026:
identical valid metrics in all 4 epochs.
Run A (4 epochs, valid every epoch) vs run B (killed once its resume file holds epoch >= 1, then
restarted with the same command). Valid metrics of every epoch must be identical. Run from the MoMoK checkout, CPU."""
import os
import re
import shutil
import subprocess
import sys
import time

import torch

SRC = "datasets/FB15K-237-branch-mkg-Y-v0"
DST = "datasets/FB15K-237-branch-tiny"
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(DST)
for f in ("entity2id.txt", "relation2id.txt", "img_features.pth", "text_features.pth"):
    shutil.copy(os.path.join(SRC, f), DST)
for f, n in (("train.txt", 1500), ("valid.txt", 150), ("test.txt", 150)):
    with open(os.path.join(SRC, f)) as fi, open(os.path.join(DST, f), "w") as fo:
        fo.writelines(fi.readlines()[:n])
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUNBUFFERED="1")
cmd = lambda resume: [sys.executable, "train.py", "--cuda", "-1", "--dataset", "FB15K-237-branch-tiny", "--epochs", "4",
                      "--eval_freq", "1", "--save", "0", "--dim", "300", "--lr", "0.0005", "--mu", "0.0001",
                      "--resume", resume]


def val_lines(text):
    out, ep = {}, None
    for line in text.splitlines():
        m = re.match(r"Epoch: (\d+)", line)
        if m:
            ep = int(m.group(1))
        elif ep is not None and line.startswith("val_"):
            out[ep] = line
            ep = None
    return out


for f in ("resA.pt", "resB.pt"):
    if os.path.exists(f):
        os.remove(f)
a = val_lines(subprocess.run(cmd("resA.pt"), env=env, capture_output=True, text=True).stdout)
log = open("runB.log", "w")
p = subprocess.Popen(cmd("resB.pt"), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
# never open resB.pt while B runs (Windows refuses os.replace onto an open file): wait for the 2nd valid line,
# then for the resume file to be rewritten and settle, then kill
t_seen = None
while True:
    time.sleep(0.5)
    if p.poll() is not None:
        raise SystemExit("run B ended before it could be interrupted")
    n_val = open("runB.log", errors="replace").read().count("val_MRR")
    if n_val >= 2 and t_seen is None:
        t_seen = time.time()
    if t_seen and os.path.exists("resB.pt") and os.path.getmtime("resB.pt") > t_seen and             time.time() - os.path.getmtime("resB.pt") > 2 and not os.path.exists("resB.pt.tmp"):
        break
p.kill()
p.wait()
log.close()
print("killed B after epoch", torch.load("resB.pt", map_location="cpu", weights_only=False)["epoch"])
first = val_lines(open("runB.log").read())
second = subprocess.run(cmd("resB.pt"), env=env, capture_output=True, text=True).stdout
assert "resumed from resB.pt" in second, second[-2000:]
b = {**first, **val_lines(second)}
print("A:", a)
print("B:", b)
same = all(a[k] == b.get(k) for k in a) and len(a) == 4
print("identical per epoch:", same)
sys.exit(0 if same else 1)
