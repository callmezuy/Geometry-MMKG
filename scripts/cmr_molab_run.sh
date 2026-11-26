#!/bin/bash
# CMR on Molab (RESEARCH_PLAN.md P3.7), restartable; started by notebooks/cmr.py.
# CMR_TASK = FB15k237_ind (default, FB15K-237-ind) or SpatialVRD_ind (Spatial-VRD).
# Needs HF_TOKEN; REPO_ROOT = this repository (default /marimo/Geometry-MMKG). Logs: /marimo/runs/cmr_<ck>/logs/progress.txt.
# - training state (resume.pt, model_best.mdl, log.txt) goes to a private state repo (one per dataset) after every
#   epoch (history squashed, unused stored files deleted); a new session restores it and continues (--resume), or
#   goes straight to evaluation when the run had finished
# - evaluation: valid (no kNN; kNN all lambdas) -> lambda on valid -> test once per variant; ranks with ties at
#   their mean (baselines/patches/CMR.diff, gm_rank)
TASK=${CMR_TASK:-FB15k237_ind}
case $TASK in
  FB15k237_ind)   DS=FB15K-237-ind; CK=fb15k237;   export CMR_RESUME_REPO=callmezuy/cmr-resume ;;
  SpatialVRD_ind) DS=Spatial-VRD;   CK=spatialvrd; export CMR_RESUME_REPO=callmezuy/cmr-resume-spatial-vrd ;;
  *) echo "unknown CMR_TASK $TASK"; exit 1 ;;
esac
RUNS=/marimo/runs/cmr_$CK
LOG=$RUNS/logs; mkdir -p $LOG
export CMR_BASE=$DS CMR_LOG=$LOG
C=/marimo/baselines/CMR
R=${REPO_ROOT:-/marimo/Geometry-MMKG}
OUT=$RUNS/eval
DATA=/marimo/data/$DS
PY=$(which python)
export HF_HUB_DISABLE_XET=1
step() { echo "START $1 $(date +%T)" >> $LOG/progress.txt; }
done_() { echo "END $1 rc=$2 $(date +%T)" >> $LOG/progress.txt; }
fail() { echo "ALLDONE failed at $1 $(date +%T)" >> $LOG/progress.txt; exit 1; }

step setup
echo "task: $TASK ($DS)" >> $LOG/progress.txt
cd /marimo
# CMR without a GPU would silently train on the CPU (about 25 h per epoch): stop instead
if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "NO GPU in this sandbox: choose a GPU machine on Molab, then run the notebook again" >> $LOG/progress.txt
  fail no_gpu
fi
($PY -m pip install -q gdown faiss-cpu || uv pip install --python $PY gdown faiss-cpu) > $LOG/pip.log 2>&1
done_ setup $?

# data and images of each dataset (our v0 files on HF; the image source checked by sha256)
get_data() {
  mkdir -p /marimo/raw
  if [ $TASK = FB15k237_ind ]; then
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='callmezuy/fb15k237-ind', repo_type='dataset', revision='v0', local_dir='$DATA', allow_patterns=['*.txt','*.tsv','ent_img_mask.pt'])" &&
    { [ -s /marimo/raw/fb15k_wn18_images.zip ] || $PY -m gdown 197c4fCLVC6F7sCBqZIDwm5tjWeGJnZ5- -O /marimo/raw/fb15k_wn18_images.zip; } &&
    echo "a778238cdef7876eda64cff89d8599c63ebb684b1efdf71ccd034ed04272bdb7  /marimo/raw/fb15k_wn18_images.zip" | sha256sum -c - &&
    ZIP=/marimo/raw/fb15k_wn18_images.zip
  else
    # VRD photos: the Kaggle VRD zip, mirrored unchanged in the private repo callmezuy/vrd-images-raw
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='callmezuy/spatial-vrd', repo_type='dataset', revision='v0', local_dir='$DATA', allow_patterns=['*.txt','sg_*_annotations.json','ent_img_mask.pt'])" &&
    $PY -c "from huggingface_hub import hf_hub_download; hf_hub_download('callmezuy/vrd-images-raw', 'visual-relationship-detection-vrd-dataset.zip', repo_type='dataset', local_dir='/marimo/raw')" &&
    echo "b858225e45b3cf911642596188c6a81499508220e188ac2a286e67ac373df210  /marimo/raw/visual-relationship-detection-vrd-dataset.zip" | sha256sum -c - &&
    ZIP=/marimo/raw/visual-relationship-detection-vrd-dataset.zip
  fi
}

step cmr_setup
{
  # Molab keeps only .py / .md files between sessions: a CMR folder without .git is a leftover, clone it again
  { [ -d $C/.git ] || rm -rf /marimo/baselines/CMR; } &&
  { [ ! -d $C/.git ] || git -C $C -c core.autocrlf=false checkout -q -- .; } &&
  $PY $R/scripts/setup_baseline.py cmr --dest $C &&
  get_data &&
  { [ -s $C/data/$TASK/img_entities.json ] || $PY $R/scripts/prepare_cmr_data.py $TASK --cmr $C --data $DATA --zip $ZIP; } &&
  $PY -c "from huggingface_hub import snapshot_download as s; s('bert-base-uncased', local_dir='$C/PLMs/bert-base-uncased', allow_patterns=['*.json','*.txt','model.safetensors']); s('google/vit-base-patch16-224', local_dir='$C/PLMs/vit-base-patch16-224', allow_patterns=['*.json','model.safetensors'])"
} > $LOG/cmr_setup.log 2>&1
rc=$?; done_ cmr_setup $rc; [ $rc -ne 0 ] && fail cmr_setup

step restore
mkdir -p $C/ckpt/$CK
read RUN STATE < <($PY $R/scripts/cmr_hf.py restore $C/ckpt/$CK 2> $LOG/restore.log | tail -1)
echo "restore: run=$RUN state=$STATE" >> $LOG/progress.txt
done_ restore $?

if [ "$STATE" = "none" ]; then
  # exact-resume check before a fresh run (CPU, 64 training triples, 3 epochs: straight vs killed + --resume)
  step resume_test
  {
    cd $C && TASK=$TASK $PY - <<'PYEOF'
import json, os
task = os.environ["TASK"]
os.makedirs("data/tiny", exist_ok=True)
for s, n in (("train", 64), ("valid", 32), ("test", 32)):
    d = json.load(open(f"data/{task}/{s}.txt.json", encoding="utf-8"))[:n]
    json.dump(d, open(f"data/tiny/{s}.txt.json", "w", encoding="utf-8"))
for f in ("relations.json", "img_entities.json"):
    open(f"data/tiny/{f}", "wb").write(open(f"data/{task}/{f}", "rb").read())
PYEOF
    $PY $R/scripts/cmr_resume_check.py $RUNS/cmr_resume_test_$(date +%s)
  } > $LOG/resume_test.log 2>&1
  rc=$?; done_ resume_test $rc; [ $rc -ne 0 ] && fail resume_test
fi

# the README command of the authors (FB15k237_ind), with the ViT frozen as in the paper; the same for Spatial-VRD
TRAIN="--pretrained-model ./PLMs/bert-base-uncased --pooling mean --lr 1e-5 --use-link-graph --task $TASK
  --batch-size 768 --print-freq 20 --additive-margin 0.0 --finetune-t --pre-batch 2 --epochs 50 --workers 12
  --max-to-keep 5 --use-amp --mm --prefix 4 --freeze-vit"
cd $C
if [ "$STATE" != "finished" ]; then
  if [ "$STATE" = "resume" ]; then
    RUN_DIR=$C/ckpt/$CK/$RUN
    step cmr_train_resume
    $PY main.py --model-dir ./ckpt/$CK/ --resume ./ckpt/$CK/$RUN $TRAIN >> $LOG/cmr_train.log 2>&1 &
  else
    step cmr_train
    $PY main.py --model-dir ./ckpt/$CK/ $TRAIN > $LOG/cmr_train.log 2>&1 &
  fi
  TPID=$!
  # the run folder appears once config.py has run
  for i in $(seq 1 120); do
    [ -n "$RUN_DIR" ] && break
    RUN_DIR=$(ls -td $C/ckpt/$CK/${TASK,,}-*/ 2>/dev/null | head -1); RUN_DIR=${RUN_DIR%/}; sleep 5
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
  RUN_DIR=$C/ckpt/$CK/$RUN
fi

# ------------------------------------------------------------------ evaluation (lambda chosen on valid)
mkdir -p $OUT
BEST=$RUN_DIR/model_best.mdl
COMMON="--task $TASK --pretrained-model ./PLMs/bert-base-uncased --batch-size 1024 --mm --prefix 4 --eval-model-path $BEST --workers 12"
step valid_noknn
$PY evaluate.py $COMMON --knn_topk 0 > $OUT/eval_valid_noknn.log 2>&1; done_ valid_noknn $?
step valid_knn
$PY evaluate.py $COMMON --knn_topk 32 > $OUT/eval_valid_knn.log 2>&1; done_ valid_knn $?
$PY - $OUT/eval_valid_knn.log $OUT/lambda.json <<'PYEOF'
# lambda = the best valid MRR of the protocol ranks (ties at their mean, the ranks that go to the rank file); CMR's own
# metric (its sort order decides ties) is kept next to it for reference
import ast, json, re, sys
lam, rows = None, []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.search(r"knn lambda: ([0-9.]+)", line)
    if m:
        lam, own = float(m.group(1)), None
    m = re.search(r"Averaged metrics text-mm: (\{.*\})", line)
    if m and lam is not None:
        own = ast.literal_eval(m.group(1))
    m = re.search(r"\[Geometry-MMKG\] protocol metrics text-mm: (\{.*\})", line)
    if m and lam is not None:
        rows.append({"lambda": lam, "protocol": json.loads(m.group(1)), "cmr_own": own})
        lam = None
assert len(rows) == 41, f"expected 41 lambdas (0, 0.025, ..., 1), found {len(rows)}"
best = max(rows, key=lambda x: x["protocol"]["mrr"])
json.dump({"chosen_lambda": best["lambda"], "valid_text_mm": best["protocol"], "rule": "ties at their mean",
           "all": rows}, open(sys.argv[2], "w"), indent=1)
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
  $PY $R/scripts/cmr_ranks.py --cmr $C --task $TASK --data $DATA \
    --ranks $OUT/ranks_cmr_test_$v.tsv --out $OUT/ranks_cmr_${DS}_${v}_test.csv &&
  $PY $R/scripts/compute_metrics.py --data $DATA \
    --ranks cmr_$v=$OUT/ranks_cmr_${DS}_${v}_test.csv --out $OUT/metrics_cmr_${v}_test.md
done > $OUT/metrics.log 2>&1; done_ metrics $?
step upload
cp $RUN_DIR/log.txt $OUT/train_log.txt 2>/dev/null
cp $C/data/$TASK/gm_prepare.json $OUT/ 2>/dev/null
$PY - <<PYEOF > $OUT/upload.log 2>&1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
dst = "cmr/$DS/$RUN"
api.upload_folder(repo_id="callmezuy/geometry-mmkg-results", repo_type="dataset", folder_path="$OUT",
                  path_in_repo=dst + "/eval", commit_message="CMR $DS: evaluation")
api.upload_file(path_or_fileobj="$BEST", path_in_repo=dst + "/model_best.mdl", repo_id="callmezuy/geometry-mmkg-results",
                repo_type="dataset", commit_message="CMR $DS: final checkpoint")
print("uploaded")
PYEOF
done_ upload $?
echo "ALLDONE $(date +%T)" >> $LOG/progress.txt
