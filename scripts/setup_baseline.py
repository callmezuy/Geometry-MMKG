"""
Fetch a baseline that is NOT kept in this repository and apply our changes to it (RESEARCH_PLAN.md P3.6, P3.7;
decided 25/11/2026). MoMoK and CMR publish no licence, so the repository holds only our diff
(baselines/patches/<name>.diff, listed in baselines/PROVENANCE.md); this script clones the authors' repository at
the pinned commit and applies the diff. Re-running it on a finished checkout only verifies it.

  python scripts/setup_baseline.py momok --dest /marimo/baselines/MoMoK
  python scripts/setup_baseline.py cmr   --dest /marimo/baselines/CMR
  python scripts/setup_baseline.py momok --dest /tmp/MoMoK-orig --no-patch     # the authors' code as released

Checks: the checkout is at the pinned commit; the patch applies to the pristine tree (git apply --check) or is
already applied (reverse check); with --no-patch the tree must be pristine.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = {
    # name: (upstream repository, pinned commit, patch in baselines/patches/)
    "momok": ("https://github.com/zjukg/MoMoK.git", "99c2df114d48c79708ea0608644b685183d81bd9", "MoMoK.diff"),
    "cmr": ("https://github.com/OreOZhao/CMR.git", "bcd23a66fb5787f5968b7615a836c8b48361b5ec", "CMR.diff"),
}


def git(dest, *args, check=True, input_bytes=None):
    # core.autocrlf=false: the patches are LF, like the upstream blobs, on every OS
    return subprocess.run(["git", "-c", "core.autocrlf=false", "-C", dest, *args], check=check,
                          input=input_bytes, capture_output=True)


def patch_bytes(name):
    with open(os.path.join(ROOT, "baselines", "patches", BASELINES[name][2]), "rb") as f:
        return f.read().replace(b"\r\n", b"\n")   # a Windows checkout of this repo may have turned it into CRLF


def setup(name, dest, apply_patch=True):
    url, commit, _ = BASELINES[name]
    dest = os.path.abspath(dest)
    if not os.path.isdir(os.path.join(dest, ".git")):
        if os.path.exists(dest) and os.listdir(dest):
            raise SystemExit(f"{dest} exists and is not a git checkout; remove it or choose another --dest")
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "-q", url, dest], check=True)
        git(dest, "checkout", "-q", commit)
    head = git(dest, "rev-parse", "HEAD").stdout.decode().strip()
    if head != commit:
        raise SystemExit(f"{dest} is at {head}, expected {commit}")
    patch = patch_bytes(name)
    pristine = git(dest, "diff", "--quiet", check=False).returncode == 0
    if not apply_patch:
        if not pristine:
            raise SystemExit(f"{dest} has local changes; --no-patch needs the authors' code unchanged")
        return f"{name}: {dest} at {commit[:7]}, authors' code (no patch)"
    if pristine:
        r = git(dest, "apply", "--check", "-", check=False, input_bytes=patch)
        if r.returncode != 0:
            raise SystemExit(f"patch does not apply to {dest}:\n{r.stderr.decode()}")
        git(dest, "apply", "-", input_bytes=patch)
        state = "patch applied"
    elif git(dest, "apply", "--reverse", "--check", "-", check=False, input_bytes=patch).returncode == 0:
        state = "patch already applied"
    else:
        raise SystemExit(f"{dest} has changes that are not exactly baselines/patches/{BASELINES[name][2]}; "
                         "reset it (git checkout -- .) and run again")
    return f"{name}: {dest} at {commit[:7]}, {state}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=sorted(BASELINES))
    ap.add_argument("--dest", required=True)
    ap.add_argument("--no-patch", action="store_true", help="only fetch the authors' code at the pinned commit")
    a = ap.parse_args(argv)
    print(setup(a.name, a.dest, apply_patch=not a.no_patch))


if __name__ == "__main__":
    sys.exit(main())
