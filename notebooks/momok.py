import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    import shutil
    import subprocess
    import marimo as mo

    mo.md(
        """
        # MoMoK trên Molab: tái lập và lần chạy config gốc
        RESEARCH_PLAN.md mục P3.6 / P4.0. MoMoK (Zhang et al., ICLR 2025) chạy **nguyên bản**: code của tác giả ở
        commit cố định cộng diff của mình (`baselines/patches/MoMoK.diff`: ma trận nhãn dựng trên GPU, chọn
        checkpoint theo valid, rank chặt, chạy tiếp khi bị ngắt; bảng thay đổi trong `baselines/PROVENANCE.md`).

        `scripts/momok_molab_run.sh` chạy lần lượt 4 lần chạy (lệnh README của MKG-Y: lr 5e-4, mu 1e-4, dim 300,
        2000 epoch):
        1. MKG-Y, đặc trưng gốc của tác giả, `--select_on original` (như code gốc: lấy giá trị tốt nhất của từng
           metric qua các lần đánh giá trên test). Đối chiếu bài báo: MRR 37,91, H@1 35,09.
        2. Như 1 nhưng `--select_on valid`: đặc trưng gốc, giao thức của mình (chọn trên valid, rank chặt).
        3. FB15k-237-IMG v0, Control, seed 42: lần chạy config gốc để chốt số epoch.
        4. MKG-Y v0 (đặc trưng của mình), Control, seed 42: như 3.

        Mỗi lần chạy lưu trạng thái lên repo private `callmezuy/momok-resume` sau mỗi lần đánh giá (100 epoch).
        Kết quả (log, rank, metric) lên `geometry-mmkg-results/momok/<dataset>/<run>/`.

        **Thứ tự dùng:** dán HF token (quyền write), nhấn Enter → bấm ▶. Sandbox Molab bị khởi động lại thì mở lại
        notebook, nhập token, bấm ▶: các lần chạy đã xong được bỏ qua, lần đang dở chạy tiếp từ lần đánh giá cuối.
        """
    )
    return mo, os, re, shutil, subprocess


@app.cell
def _(os, shutil, subprocess):
    # Mã nguồn Geometry-MMKG (một lần cho cả session)
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "baselines", "patches", "MoMoK.diff")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "baselines", "patches", "MoMoK.diff")):
        REPO_ROOT = os.path.dirname(WORK_DIR)
    else:
        REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
        # Molab chỉ giữ file .py/.md giữa các session (mất .git), nên thiếu .git thì clone lại.
        if not os.path.isdir(os.path.join(REPO_ROOT, ".git")):
            shutil.rmtree(REPO_ROOT, ignore_errors=True)
            subprocess.run(["git", "clone", "https://github.com/callmezuy/Geometry-MMKG.git", REPO_ROOT], check=True)
        else:
            subprocess.run(["git", "-C", REPO_ROOT, "pull"], check=False)
    CODE_COMMIT = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"], text=True, capture_output=True
    ).stdout.strip()
    print(f"Geometry-MMKG code at commit {CODE_COMMIT}")
    return CODE_COMMIT, REPO_ROOT


@app.cell
def _(mo):
    hf_token_ui = mo.ui.text(kind="password", label="HF token (quyền write)", full_width=True)
    start_btn = mo.ui.run_button(label="▶ Chạy / chạy tiếp MoMoK")
    mo.vstack([hf_token_ui, start_btn])
    return hf_token_ui, start_btn


@app.cell
def _(REPO_ROOT, hf_token_ui, mo, os, start_btn, subprocess):
    # Khởi chạy scripts/momok_molab_run.sh ở nền: không chạy dài trong cell (luồng của cell có thể bị cắt).
    # Token chỉ nằm trong biến môi trường của tiến trình đó.
    mo.stop(not start_btn.value, mo.md("⏸ Nhập token rồi bấm ▶."))
    _tok = hf_token_ui.value.strip()
    mo.stop(not _tok, mo.md("⚠ Chưa nhập HF token: nhập, nhấn Enter, rồi bấm ▶ lại."))
    _runs = "/marimo/runs/momok"
    os.makedirs(_runs, exist_ok=True)
    _pidf = _runs + "/run_momok.pid"
    if os.path.exists(_pidf) and os.path.exists("/proc/" + open(_pidf).read().strip()):
        _msg = "MoMoK đang chạy; xem tiến độ bên dưới."
    else:
        _p = subprocess.Popen(
            ["bash", os.path.join(REPO_ROOT, "scripts", "momok_molab_run.sh")],
            env={**os.environ, "HF_TOKEN": _tok, "REPO_ROOT": REPO_ROOT},
            start_new_session=True, stdout=open(_runs + "/run_momok.out", "a"), stderr=subprocess.STDOUT,
        )
        open(_pidf, "w").write(str(_p.pid))
        _msg = f"Đã khởi chạy MoMoK (pid {_p.pid})."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["1m", "5m"], default_interval="1m", label="Làm mới tiến độ")
    refresh
    return (refresh,)


@app.cell
def _(mo, os, re, refresh):
    # Tiến độ: các bước của script; mỗi lần chạy: epoch hiện tại, các lần đánh giá, kết quả test; sao lưu
    refresh
    _root = "/marimo/runs/momok"
    _steps = open(f"{_root}/progress.txt").read() if os.path.exists(f"{_root}/progress.txt") else "(chưa chạy)"
    _parts = []
    for _run in sorted(d for d in os.listdir(_root) if os.path.isdir(os.path.join(_root, d))) if os.path.isdir(_root) else []:
        _f = os.path.join(_root, _run, "train.log")
        if not os.path.exists(_f):
            continue
        _txt = open(_f, encoding="utf-8", errors="replace").read()
        _ep = re.findall(r"(\d+)/2000 \[", _txt)
        _evals = re.findall(r"Epoch: (\d+)\n((?:val|test)_[^\n]*)", _txt)
        _lines = [f"epoch {e}: {m[:160]}" for e, m in _evals[-4:]]
        _res = _txt[_txt.rfind("Test set results:"):].splitlines()[:2] if "Test set results:" in _txt else []
        _parts.append(f"**{_run}** (epoch {_ep[-1] if _ep else '?'} / 2000)\n```\n" + "\n".join(_lines + _res) + "\n```")
    _backup = open(f"{_root}/backup.log").read().splitlines()[-3:] if os.path.exists(f"{_root}/backup.log") else []
    mo.md(
        "### Các bước\n```\n" + _steps + "\n```\n### Các lần chạy\n" + "\n\n".join(_parts)
        + ("\n### Sao lưu\n```\n" + "\n".join(_backup) + "\n```\n" if _backup else "")
    )
    return


if __name__ == "__main__":
    app.run()
