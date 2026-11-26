#!/bin/bash
# CMR on FB15K-237-ind on Molab (RESEARCH_PLAN.md P3.7), restartable; started by notebooks/cmr.py.
# Needs HF_TOKEN; REPO_ROOT = this repository (default /marimo/Geometry-MMKG). Logs: /marimo/runs/logs/progress.txt.
# - training state (resume.pt, model_best.mdl, log.txt) goes to the private repo callmezuy/cmr-resume after every
#   epoch (history squashed); a new session restores it and continues (--resume), or goes straight to evaluation
#   when the run had finished
# - evaluation as before: valid (no kNN; kNN all lambdas) -> lambda on valid -> test once per variant
LOG=/marimo/runs/logs; mkdir -p $LOG
RUNS=/marimo/runs
C=/marimo/baselines/CMR
R=${REPO_ROOT:-/marimo/Geometry-MMKG}
OUT=$RUNS/cmr_eval
PY=$(which python)
export HF_HUB_DISABLE_XET=1
step() { echo "START $1 $(date +%T)" >> $LOG/progress.txt; }
done_() { echo "END $1 rc=$2 $(date +%T)" >> $LOG/progress.txt; }
fail() { echo "ALLDONE failed at $1 $(date +%T)" >> $LOG/progress.txt; exit 1; }

step setup
cd /marimo
# CMR without a GPU would silently train on the CPU (about 25 h per epoch): stop instead
if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "NO GPU in this sandbox: choose a GPU machine on Molab, then run the notebook again" >> $LOG/progress.txt
  fail no_gpu
fi
($PY -m pip install -q gdown faiss-cpu || uv pip install --python $PY gdown faiss-cpu) > $LOG/pip.log 2>&1
done_ setup $?

step cmr_setup
{
  # Molab keeps only .py / .md files between sessions: a CMR folder without .git is a leftover, clone it again
  { [ -d $C/.git ] || rm -rf /marimo/baselines/CMR; } &&
  { [ ! -d $C/.git ] || git -C $C -c core.autocrlf=false checkout -q -- .; } &&
  $PY $R/scripts/setup_baseline.py cmr --dest $C &&
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='callmezuy/fb15k237-ind', repo_type='dataset', revision='v0', local_dir='/marimo/data/FB15K-237-ind', allow_patterns=['*.txt','*.tsv','ent_img_mask.pt'])" &&
  mkdir -p /marimo/raw &&
  { [ -s /marimo/raw/fb15k_wn18_images.zip ] || $PY -m gdown 197c4fCLVC6F7sCBqZIDwm5tjWeGJnZ5- -O /marimo/raw/fb15k_wn18_images.zip; } &&
  echo "a778238cdef7876eda64cff89d8599c63ebb684b1efdf71ccd034ed04272bdb7  /marimo/raw/fb15k_wn18_images.zip" | sha256sum -c - &&
  { [ -s $C/data/FB15k237_ind/img_entities.json ] || $PY $R/scripts/prepare_cmr_data.py FB15k237_ind --cmr $C --data /marimo/data/FB15K-237-ind --zip /marimo/raw/fb15k_wn18_images.zip; } &&
  $PY -c "from huggingface_hub import snapshot_download as s; s('bert-base-uncased', local_dir='$C/PLMs/bert-base-uncased', allow_patterns=['*.json','*.txt','model.safetensors']); s('google/vit-base-patch16-224', local_dir='$C/PLMs/vit-base-patch16-224', allow_patterns=['*.json','model.safetensors'])"
} > $LOG/cmr_setup.log 2>&1
rc=$?; done_ cmr_setup $rc; [ $rc -ne 0 ] && fail cmr_setup

step restore
mkdir -p $C/ckpt/fb15k237
read RUN STATE < <($PY $R/scripts/cmr_hf.py restore $C/ckpt/fb15k237 2> $LOG/restore.log | tail -1)
echo "restore: run=$RUN state=$STATE" >> $LOG/progress.txt
done_ restore $?

if [ "$STATE" = "none" ]; then
  # exact-resume check before a fresh run (CPU, 64 training triples, 3 epochs: straight vs killed + --resume)
  step resume_test
  {
    cd $C && $PY - <<'PYEOF'
import json, os
os.makedirs("data/tiny", exist_ok=True)
for s, n in (("train", 64), ("valid", 32), ("test", 32)):
    d = json.load(open(f"data/FB15k237_ind/{s}.txt.json", encoding="utf-8"))[:n]
    json.dump(d, open(f"data/tiny/{s}.txt.json", "w", encoding="utf-8"))
for f in ("relations.json", "img_entities.json"):
    open(f"data/tiny/{f}", "wb").write(open(f"data/FB15k237_ind/{f}", "rb").read())
PYEOF
    $PY $R/scripts/cmr_resume_check.py $RUNS/cmr_resume_test_$(date +%s)
  } > $LOG/resume_test.log 2>&1
  rc=$?; done_ resume_test $rc; [ $rc -ne 0 ] && fail resume_test
fi

TRAIN="--pretrained-model ./PLMs/bert-base-uncased --pooling mean --lr 1e-5 --use-link-graph --task FB15k237_ind
  --batch-size 768 --print-freq 20 --additive-margin 0.0 --finetune-t --pre-batch 2 --epochs 50 --workers 12
  --max-to-keep 5 --use-amp --mm --prefix 4 --freeze-vit"
cd $C
if [ "$STATE" != "finished" ]; then
  if [ "$STATE" = "resume" ]; then
    RUN_DIR=$C/ckpt/fb15k237/$RUN
    step cmr_train_resume
    $PY main.py --model-dir ./ckpt/fb15k237/ --resume ./ckpt/fb15k237/$RUN $TRAIN >> $LOG/cmr_train.log 2>&1 &
  else
    step cmr_train
    $PY main.py --model-dir ./ckpt/fb15k237/ $TRAIN > $LOG/cmr_train.log 2>&1 &
  fi
  TPID=$!
  # the run folder appears once config.py has run
  for i in $(seq 1 120); do
    [ -n "$RUN_DIR" ] && break
    RUN_DIR=$(ls -td $C/ckpt/fb15k237/fb15k237_ind-*/ 2>/dev/null | head -1); RUN_DIR=${RUN_DIR%/}; sleep 5
  done
  [ -n "$RUN_DIR" ] || fail no_run_dir
  RUN=$(basename "$RUN_DIR")   # a fresh run gets its name here (restore printed "none")
  echo "run: $RUN" >> $LOG/progress.txt
  rm -f $RUNS/stop_backup
  $PY $R/scripts/cmr_hf.py backup $RUN_DIR $RUNS/stop_backup >> $LOG/backup.out 2>&1 &
  BPID=$!
  wait $TPID; trc=$?
  touch $RUNS/stop_backup; wait $BPID
  done_ cmr_train $trc
  [ $trc -ne 0 ] && fail cmr_train
  $PY $R/scripts/cmr_hf.py finished $RUN_DIR >> $LOG/backup.out 2>&1
else
  RUN_DIR=$C/ckpt/fb15k237/$RUN
fi

# ------------------------------------------------------------------ evaluation (lambda chosen on valid)
mkdir -p $OUT
BEST=$RUN_DIR/model_best.mdl
COMMON="--task FB15k237_ind --pretrained-model ./PLMs/bert-base-uncased --batch-size 1024 --mm --prefix 4 --eval-model-path $BEST --workers 12"
step valid_noknn
$PY evaluate.py $COMMON --knn_topk 0 > $OUT/eval_valid_noknn.log 2>&1; done_ valid_noknn $?
step valid_knn
$PY evaluate.py $COMMON --knn_topk 32 > $OUT/eval_valid_knn.log 2>&1; done_ valid_knn $?
$PY - $OUT/eval_valid_knn.log $OUT/lambda.json <<'PYEOF'
import ast, json, re, sys
lam, rows = None, []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.search(r"knn lambda: ([0-9.]+)", line)
    if m:
        lam = float(m.group(1))
    m = re.search(r"Averaged metrics text-mm: (\{.*\})", line)
    if m and lam is not None:
        rows.append((lam, ast.literal_eval(m.group(1))))
        lam = None
best = max(rows, key=lambda x: x[1]["mrr"])
json.dump({"chosen_lambda": best[0], "valid_text_mm": best[1], "all": rows}, open(sys.argv[2], "w"), indent=1)
PYEOF
LAM=$($PY -c "import json; print(json.load(open('$OUT/lambda.json'))['chosen_lambda'])")
echo "lambda chosen on valid: $LAM" >> $LOG/progress.txt
step test_noknn
$PY evaluate.py $COMMON --is-test --knn_topk 0 --rank-file $OUT/ranks_cmr_test_noknn.tsv --rank-tensor mm \
  > $OUT/eval_test_noknn.log 2>&1; done_ test_noknn $?
rm -f $RUN_DIR/hr_index.index $RUN_DIR/faissid2entityid.pickle   # memory built as released at test time
step test_knn
$PY evaluate.py $COMMON --is-test --knn_topk 32 --rank-lambda $LAM --rank-file $OUT/ranks_cmr_test_knn.tsv \
  --rank-tensor mm > $OUT/eval_test_knn.log 2>&1; done_ test_knn $?
step metrics
for v in noknn knn; do
  $PY $R/scripts/cmr_ranks.py --cmr $C --task FB15k237_ind --data /marimo/data/FB15K-237-ind \
    --ranks $OUT/ranks_cmr_test_$v.tsv --out $OUT/ranks_cmr_FB15K-237-ind_${v}_test.csv &&
  $PY $R/scripts/compute_metrics.py --data /marimo/data/FB15K-237-ind \
    --ranks cmr_$v=$OUT/ranks_cmr_FB15K-237-ind_${v}_test.csv --out $OUT/metrics_cmr_${v}_test.md
done > $OUT/metrics.log 2>&1; done_ metrics $?
step upload
cp $RUN_DIR/log.txt $OUT/train_log.txt 2>/dev/null
$PY - <<PYEOF > $OUT/upload.log 2>&1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
dst = "cmr/FB15K-237-ind/$RUN"
api.upload_folder(repo_id="callmezuy/geometry-mmkg-results", repo_type="dataset", folder_path="$OUT",
                  path_in_repo=dst + "/eval", commit_message="CMR FB15K-237-ind: evaluation")
api.upload_file(path_or_fileobj="$BEST", path_in_repo=dst + "/model_best.mdl", repo_id="callmezuy/geometry-mmkg-results",
                repo_type="dataset", commit_message="CMR FB15K-237-ind: final checkpoint")
print("uploaded")
PYEOF
done_ upload $?
echo "ALLDONE $(date +%T)" >> $LOG/progress.txt
