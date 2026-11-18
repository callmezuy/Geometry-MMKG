"""
Publish data/<folder>/ to a Hugging Face dataset repo (RESEARCH_PLAN.md P0.8, data v0).

Uploads every file of the folder (skips .git*, temp files starting with "_") plus hf_cards/<repo>.md as README.md
in one commit, then optionally tags it. Repos are public.

Usage (from Code/):
  python scripts/publish_hf.py --dry-run                     # list what would be uploaded
  python scripts/publish_hf.py --only fb15k237-img --tag v0
  python scripts/publish_hf.py --only spatial-vg             # VG: structure first, tag after the features land
"""
import argparse
import os

from huggingface_hub import CommitOperationAdd, HfApi

OWNER = "callmezuy"
REPOS = {  # repo name -> local folder
    "fb15k237-img": "FB15k-237-IMG",
    "fb15k237-ind": "FB15K-237-ind",
    "mkg-y": "MKG-Y",
    "spatial-vrd": "Spatial-VRD",
    "spatial-vrd-trans": "Spatial-VRD-Trans",
    "spatial-vg": "Spatial-VG",
}


def files_of(folder):
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if name.startswith((".git", "_")) or not os.path.isfile(p):
            continue
        out.append((name, p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(REPOS), help="repos to publish (default: all)")
    ap.add_argument("--tag", help="tag to create on the new commit (e.g. v0)")
    ap.add_argument("--message", default="Data v0 (RESEARCH_PLAN.md P0.8)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    for repo in a.only or list(REPOS):
        folder = os.path.join("data", REPOS[repo])
        card = os.path.join("hf_cards", f"{repo}.md")
        assert os.path.isfile(card), card
        files = files_of(folder)
        size = sum(os.path.getsize(p) for _, p in files)
        print(f"{OWNER}/{repo} <- {folder}: {len(files)} files + README.md, {size / 1e9:.2f} GB")
        for name, p in files:
            print(f"    {name:32s} {os.path.getsize(p):>13,}")
        if a.dry_run:
            continue
        repo_id = f"{OWNER}/{repo}"
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card)]
        ops += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=p) for name, p in files]
        info = api.create_commit(repo_id, repo_type="dataset", operations=ops, commit_message=a.message)
        print(f"    commit {info.oid}")
        if a.tag:
            api.create_tag(repo_id, repo_type="dataset", tag=a.tag, revision=info.oid)
            print(f"    tag {a.tag}")


if __name__ == "__main__":
    main()
