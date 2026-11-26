"""HF helper for the CMR runs on Molab (scripts/cmr_molab_run.sh, RESEARCH_PLAN.md P3.7) (sessions can end at any
time). Token: HF_TOKEN in the environment; CMR_RESUME_REPO / CMR_BASE / CMR_LOG choose the dataset (defaults:
FB15K-237-ind). One state repo per dataset, so two runs never squash or clean the same repo.
State repo (private, only the latest version kept: history squashed and unused stored files deleted after every
upload):
  <repo>  <base>/current_run.txt, <base>/<run>/{resume.pt, model_best.mdl, log.txt},
          <base>/<run>/FINISHED (training over)
  e.g. callmezuy/cmr-resume with base FB15K-237-ind; callmezuy/cmr-resume-spatial-vrd with base Spatial-VRD
  python cmr_hf.py restore <ckpt_root>            -> prints "<run> <none|resume|finished>", downloads the files
  python cmr_hf.py backup <run_dir> <stop_file>   -> loop: upload after every new resume.pt / model_best.mdl
  python cmr_hf.py finished <run_dir>             -> final upload + FINISHED marker
"""
import os
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("CMR_RESUME_REPO", "callmezuy/cmr-resume")
BASE = os.environ.get("CMR_BASE", "FB15K-237-ind")
LOG = os.environ.get("CMR_LOG", "/marimo/runs/logs")
api = HfApi(token=os.environ["HF_TOKEN"])


def files():
    try:
        return set(api.list_repo_files(REPO, repo_type="dataset"))
    except Exception:
        return set()


def restore(root):
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    fs = files()
    if f"{BASE}/current_run.txt" not in fs:
        print("none none")
        return
    run = open(hf_hub_download(REPO, f"{BASE}/current_run.txt", repo_type="dataset")).read().strip()
    state = "finished" if f"{BASE}/{run}/FINISHED" in fs else ("resume" if f"{BASE}/{run}/resume.pt" in fs else "none")
    if state == "none":
        print(f"{run} none")
        return
    d = os.path.join(root, run)
    os.makedirs(d, exist_ok=True)
    wanted = ["model_best.mdl", "log.txt"] + (["resume.pt"] if state == "resume" else [])
    for f in wanted:
        if f"{BASE}/{run}/{f}" in fs:
            p = hf_hub_download(REPO, f"{BASE}/{run}/{f}", repo_type="dataset", local_dir=d + "/_dl")
            os.replace(p, os.path.join(d, f))
    print(f"{run} {state}")


def upload(run_dir, names, message):
    run = os.path.basename(os.path.normpath(run_dir))
    for n in names:
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{BASE}/{run}/{n}", repo_id=REPO, repo_type="dataset",
                            commit_message=message)
    tmp = os.path.join(run_dir, "_current_run.txt")
    open(tmp, "w").write(run)
    api.upload_file(path_or_fileobj=tmp, path_in_repo=f"{BASE}/current_run.txt", repo_id=REPO, repo_type="dataset")
    api.super_squash_history(repo_id=REPO, repo_type="dataset")   # keep only the latest version (3+ GB each)
    prune()


def prune():
    """The squash drops the old commits but HF still stores (and bills) their large files: delete every stored file
    that the current version does not use. Only files pushed before the latest commit are touched."""
    try:
        last = api.list_repo_commits(REPO, repo_type="dataset")[0].created_at
        live = {s.lfs.sha256 for s in api.dataset_info(REPO, files_metadata=True).siblings if s.lfs}
        old = [f for f in api.list_lfs_files(REPO, repo_type="dataset") if f.file_oid not in live and f.pushed_at < last]
        if old:
            api.permanently_delete_lfs_files(REPO, old, repo_type="dataset", rewrite_history=False)
    except Exception as e:   # a failed clean-up must not stop the backups; the next upload tries again
        print(f"prune failed: {e!r}", file=sys.stderr)


def backup(run_dir, stop):
    seen = {}
    log = open(os.path.join(LOG, "backup.log"), "a", buffering=1)
    while not os.path.exists(stop):
        res = os.path.join(run_dir, "resume.pt")
        try:
            if os.path.exists(res) and not os.path.exists(res + ".tmp") and time.time() - os.path.getmtime(res) > 30:
                m = os.path.getmtime(res)
                if seen.get("resume") != m:
                    upload(run_dir, ["resume.pt", "model_best.mdl", "log.txt"], f"CMR {BASE}: resume state")
                    seen["resume"] = m
                    log.write(f"{time.strftime('%H:%M:%S')} uploaded resume state (mtime {m})\n")
        except Exception as e:
            log.write(f"{time.strftime('%H:%M:%S')} upload failed: {e!r}\n")
        time.sleep(60)


def finished(run_dir):
    upload(run_dir, ["model_best.mdl", "log.txt"], f"CMR {BASE}: training finished")
    run = os.path.basename(os.path.normpath(run_dir))
    api.upload_file(path_or_fileobj=b"done\n", path_in_repo=f"{BASE}/{run}/FINISHED", repo_id=REPO,
                    repo_type="dataset")
    try:
        api.delete_file(f"{BASE}/{run}/resume.pt", repo_id=REPO, repo_type="dataset")
    except Exception:
        pass
    api.super_squash_history(repo_id=REPO, repo_type="dataset")
    prune()


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"restore": lambda: restore(sys.argv[2]), "backup": lambda: backup(sys.argv[2], sys.argv[3]),
     "finished": lambda: finished(sys.argv[2])}[cmd]()
