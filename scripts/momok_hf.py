"""HF helper for the MoMoK runs on Molab (scripts/momok_molab_run.sh, RESEARCH_PLAN.md P3.6) (sessions can end at any
time). Token: HF_TOKEN in the environment.
State repo (private, only the latest version kept: history squashed and unused stored files deleted after every
upload), one folder per run of the queue:
  callmezuy/momok-resume  <run>/resume.pt (the patch's --resume file), <run>/train.log, <run>/FINISHED
  python momok_hf.py status <run>                 -> prints none | resume | finished
  python momok_hf.py restore <run> <dir>          -> downloads <run>/resume.pt and train.log into <dir>
  python momok_hf.py backup <run> <dir> <stop>    -> loop: upload <dir>/resume.pt after every new version
  python momok_hf.py finished <run> <dir>         -> train.log + FINISHED marker, resume.pt deleted
"""
import os
import sys
import time

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

REPO = "callmezuy/momok-resume"
api = HfApi(token=os.environ["HF_TOKEN"])


def files():
    try:
        return set(api.list_repo_files(REPO, repo_type="dataset"))
    except Exception:
        return set()


def status(run):
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    fs = files()
    print("finished" if f"{run}/FINISHED" in fs else ("resume" if f"{run}/resume.pt" in fs else "none"))


def restore(run, d):
    os.makedirs(d, exist_ok=True)
    for f in ("resume.pt", "train.log"):
        if f"{run}/{f}" in files():
            p = hf_hub_download(REPO, f"{run}/{f}", repo_type="dataset", local_dir=os.path.join(d, "_dl"))
            os.replace(p, os.path.join(d, f))
    print("restored", run)


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


def commit(ops, message):
    api.create_commit(REPO, ops, repo_type="dataset", commit_message=message)
    api.super_squash_history(repo_id=REPO, repo_type="dataset")   # keep only the latest version (about 1.6 GB per run)
    prune()


def backup(run, d, stop):
    """Runs beside one training run and is stopped (stop file) before finished(), so no two uploads overlap."""
    seen = os.path.getmtime(os.path.join(d, "resume.pt")) if os.path.exists(os.path.join(d, "resume.pt")) else None
    log = open(os.path.join(os.path.dirname(os.path.normpath(d)), "backup.log"), "a", buffering=1)
    res = os.path.join(d, "resume.pt")
    while not os.path.exists(stop):
        try:
            if os.path.exists(res) and not os.path.exists(res + ".tmp") and time.time() - os.path.getmtime(res) > 30:
                m = os.path.getmtime(res)
                if seen != m:   # a restored resume.pt is already on HF
                    ops = [CommitOperationAdd(f"{run}/resume.pt", res)]
                    if os.path.exists(os.path.join(d, "train.log")):
                        ops.append(CommitOperationAdd(f"{run}/train.log", os.path.join(d, "train.log")))
                    commit(ops, f"MoMoK {run}: resume state")
                    seen = m
                    log.write(f"{time.strftime('%H:%M:%S')} {run}: uploaded resume state (mtime {m})\n")
        except Exception as e:
            log.write(f"{time.strftime('%H:%M:%S')} {run}: upload failed: {e!r}\n")
        time.sleep(60)


def finished(run, d):
    ops = [CommitOperationAdd(f"{run}/FINISHED", b"done\n")]
    if os.path.exists(os.path.join(d, "train.log")):
        ops.append(CommitOperationAdd(f"{run}/train.log", os.path.join(d, "train.log")))
    if f"{run}/resume.pt" in files():
        ops.append(CommitOperationDelete(f"{run}/resume.pt"))
    commit(ops, f"MoMoK {run}: training finished")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"status": lambda: status(sys.argv[2]), "restore": lambda: restore(sys.argv[2], sys.argv[3]),
     "backup": lambda: backup(sys.argv[2], sys.argv[3], sys.argv[4]), "finished": lambda: finished(sys.argv[2], sys.argv[3])}[cmd]()
