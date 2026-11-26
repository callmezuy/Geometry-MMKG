#!/bin/bash
# MoMoK on Molab (RESEARCH_PLAN.md P3.6 / P4.0), restartable; started by notebooks/momok.py.
# Needs HF_TOKEN; REPO_ROOT = this repository (default /marimo/Geometry-MMKG). Logs: /marimo/runs/momok/progress.txt.
# A queue of runs, one after the other. Each run keeps its --resume file on the private repo callmezuy/momok-resume
# (scripts/momok_hf.py; written after every evaluation, i.e. every 100 epochs); a new session skips finished runs
# and continues an interrupted one from its last evaluation.
#   1. MKG-Y, the authors' features, the README command, --select_on original: the reproduction check (criterion 1;
#      paper Table 1: MRR 37.91, H@1 35.09, reported as the best value of each metric over the test evaluations)
#   2. the same with --select_on valid: the authors' features under our protocol (valid selection, strict ranks)
#   3. FB15k-237-IMG v0 Control, the README MKG-Y config, seed 42: the author-config run that fixes the epoch budget
#      (P4.0), and the seed-42 tune point of that config (c1)
#   4. MKG-Y v0 Control (our features), the same
RUNS=/marimo/runs/momok
LOG=$RUNS
M=/marimo/baselines/MoMoK
R=${REPO_ROOT:-/marimo/Geometry-MMKG}
PY=$(which python)
export HF_HUB_DISABLE_XET=1
mkdir -p $RUNS
step() { echo "START $1 $(date +%T)" >> $LOG/progress.txt; }
done_() { echo "END $1 rc=$2 $(date +%T)" >> $LOG/progress.txt; }
fail() { echo "ALLDONE failed at $1 $(date +%T)" >> $LOG/progress.txt; exit 1; }

step setup
if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "NO GPU in this sandbox: choose a GPU machine on Molab, then run the notebook again" >> $LOG/progress.txt
  fail no_gpu
fi
($PY -m pip install -q gdown scikit-learn tqdm || uv pip install --python $PY gdown scikit-learn tqdm) > $LOG/pip.log 2>&1
done_ setup $?

step momok_setup
{
  # Molab keeps only .py / .md files between sessions: a MoMoK folder without .git is a leftover, clone it again
  { [ -d $M/.git ] || rm -rf /marimo/baselines/MoMoK; } &&
  { [ ! -d $M/.git ] || git -C $M -c core.autocrlf=false checkout -q -- .; } &&
  $PY $R/scripts/setup_baseline.py momok --dest $M &&
  # the authors' features (README Google Drive link), checked by sha256; only MKG-Y is used
  mkdir -p /marimo/raw &&
  { [ -s /marimo/raw/momok_features.zip ] || $PY -m gdown 1dKJdJunb11kDtFr5NLfPlFknS7cRdm9W -O /marimo/raw/momok_features.zip; } &&
  echo "215df99ce5b9c9399afb07998a648a542ca0002e708cf0507ca0e60734f7e4da  /marimo/raw/momok_features.zip" | sha256sum -c - &&
  $PY -c "
import zipfile
z = zipfile.ZipFile('/marimo/raw/momok_features.zip')
for f in ('img_features.pth', 'text_features.pth'):
    open('$M/datasets/MKG-Y/' + f, 'wb').write(z.read('embeddings/MKG-Y/' + f))
" &&
  mkdir -p $M/checkpoint/MKG-Y &&
  # our v0 data, Control image table (scripts/prepare_momok_data.py: name with FB15K-237, the authors' 768/768 branch)
  for ds in fb15k237-img:FB15k-237-IMG mkg-y:MKG-Y; do
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='callmezuy/${ds%%:*}', repo_type='dataset', revision='v0', local_dir='/marimo/data/${ds##*:}', allow_patterns=['*2id.txt','ent_rgb_emb.pt','ent_text_emb.pt','relation_category.tsv'])" &&
    $PY $R/scripts/prepare_momok_data.py --data /marimo/data/${ds##*:} --momok $M --visual rgb || exit 1
  done
} > $LOG/momok_setup.log 2>&1
rc=$?; done_ momok_setup $rc; [ $rc -ne 0 ] && fail momok_setup

# name | MoMoK --dataset | arguments | our dataset folder for the metrics
AUTHOR="--lr 0.0005 --mu 0.0001 --dim 300 --epochs 2000"   # README, MKG-Y
QUEUE=(
  "MKG-Y_authors-features_select-original|MKG-Y|$AUTHOR --select_on original|"
  "MKG-Y_authors-features_select-valid|MKG-Y|$AUTHOR --select_on valid|/marimo/data/MKG-Y"
  "FB15k-237-IMG_control_author-config_seed42|FB15K-237-branch-FB15k-237-IMG-v0|$AUTHOR --select_on valid --seed 42|/marimo/data/FB15k-237-IMG"
  "MKG-Y_control_author-config_seed42|FB15K-237-branch-mkg-Y-v0|$AUTHOR --select_on valid --seed 42|/marimo/data/MKG-Y"
)

cd $M
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r NAME DSET ARGS DATA <<< "$entry"
  D=$RUNS/$NAME
  ST=$($PY $R/scripts/momok_hf.py status $NAME 2>> $LOG/hf.log | tail -1)
  echo "$NAME: state $ST" >> $LOG/progress.txt
  [ "$ST" = "finished" ] && continue
  mkdir -p $D
  [ "$ST" = "resume" ] && $PY $R/scripts/momok_hf.py restore $NAME $D >> $LOG/hf.log 2>&1
  step $NAME
  RANKS=""; [ -n "$DATA" ] && RANKS="--rank_file $D/ranks_test.csv"
  $PY train.py --cuda 0 --dataset $DSET $ARGS --resume $D/resume.pt $RANKS >> $D/train.log 2>&1 &
  TPID=$!
  rm -f $D/stop_backup
  $PY $R/scripts/momok_hf.py backup $NAME $D $D/stop_backup >> $LOG/hf.log 2>&1 &
  BPID=$!
  wait $TPID; trc=$?
  touch $D/stop_backup; wait $BPID
  done_ $NAME $trc
  [ $trc -ne 0 ] && fail $NAME
  if [ -n "$DATA" ]; then
    $PY $R/scripts/compute_metrics.py --data $DATA --ranks momok=$D/ranks_test.csv --out $D/metrics_test.md \
      > $D/metrics.log 2>&1
  fi
  # results: the training log (the metrics MoMoK prints), strict test ranks and metrics; the resume file stays out
  $PY - <<PYEOF >> $LOG/hf.log 2>&1
import os
from huggingface_hub import HfApi
ds = {"MKG-Y": "MKG-Y-sanity", "FB15K-237-branch-FB15k-237-IMG-v0": "FB15k-237-IMG",
      "FB15K-237-branch-mkg-Y-v0": "MKG-Y"}["$DSET"]
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id="callmezuy/geometry-mmkg-results", repo_type="dataset", folder_path="$D",
    path_in_repo=f"momok/{ds}/$NAME", allow_patterns=["train.log", "ranks_test.csv", "metrics_test.md", "metrics.log"],
    commit_message="MoMoK $NAME")
print("uploaded $NAME")
PYEOF
  urc=$?; [ $urc -ne 0 ] && fail upload_$NAME
  $PY $R/scripts/momok_hf.py finished $NAME $D >> $LOG/hf.log 2>&1
  echo "$NAME: uploaded" >> $LOG/progress.txt
done
echo "ALLDONE $(date +%T)" >> $LOG/progress.txt
