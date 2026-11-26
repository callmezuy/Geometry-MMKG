import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import json
    import os
    import shutil
    import subprocess
    import marimo as mo

    mo.md(
        """
        # CMR trên Molab (FB15K-237-ind, Spatial-VRD)
        RESEARCH_PLAN.md mục P3.7. CMR (Zhao et al., SIGIR 2024) chạy **nguyên bản**: code của tác giả ở commit cố định
        cộng diff của mình (`baselines/patches/CMR.diff`, bảng thay đổi trong `baselines/PROVENANCE.md`), ViT đóng
        băng như bài báo, thực thể thiếu ảnh dùng chung một ảnh nhiễu. Hai dataset dùng cùng lệnh train của README.

        `scripts/cmr_molab_run.sh` làm lần lượt:
        1. Cài đặt, tải code CMR và áp diff, chuẩn bị dữ liệu (`scripts/prepare_cmr_data.py`), tải BERT / ViT.
        2. Nếu repo private lưu trạng thái của dataset đó (`callmezuy/cmr-resume` cho FB15K-237-ind,
           `callmezuy/cmr-resume-spatial-vrd` cho Spatial-VRD) có lần chạy trước: tải về và **train tiếp** (hoặc sang
           thẳng bước đánh giá nếu đã train xong). Nếu chưa có: kiểm tra chạy tiếp trên CPU (bản bị ngắt phải cho kết
           quả valid giống hệt bản chạy liền), rồi train từ đầu.
        3. Train; sau mỗi epoch đẩy trạng thái lên repo lưu trạng thái.
        4. Đánh giá: valid (không kNN; kNN với mọi λ) → chọn λ trên valid → test đúng một lần cho mỗi bản;
           rank, metric và checkpoint cuối lên `geometry-mmkg-results/cmr/<dataset>/<run>/`.

        **Thứ tự dùng:** chọn dataset, dán HF token (quyền write), nhấn Enter → bấm ▶. Sandbox Molab bị khởi động lại
        thì mở lại notebook, chọn đúng dataset, nhập token, bấm ▶: lần chạy tiếp tục từ epoch đã lưu.
        """
    )
    return glob, json, mo, os, shutil, subprocess


@app.cell
def _(os, shutil, subprocess):
    # Mã nguồn Geometry-MMKG (một lần cho cả session)
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "baselines", "patches", "CMR.diff")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "baselines", "patches", "CMR.diff")):
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
    # dataset -> (task của CMR, thư mục ckpt / log)
    TASKS = {"FB15K-237-ind": ("FB15k237_ind", "fb15k237"), "Spatial-VRD": ("SpatialVRD_ind", "spatialvrd")}
    dataset_ui = mo.ui.dropdown(options=list(TASKS), value="Spatial-VRD", label="Dataset")
    hf_token_ui = mo.ui.text(kind="password", label="HF token (quyền write)", full_width=True)
    start_btn = mo.ui.run_button(label="▶ Chạy / chạy tiếp CMR")
    mo.vstack([dataset_ui, hf_token_ui, start_btn])
    return TASKS, dataset_ui, hf_token_ui, start_btn


@app.cell
def _(REPO_ROOT, TASKS, dataset_ui, hf_token_ui, mo, os, start_btn, subprocess):
    # Khởi chạy scripts/cmr_molab_run.sh ở nền: không chạy dài trong cell (luồng của cell có thể bị cắt).
    # Token chỉ nằm trong biến môi trường của tiến trình đó.
    mo.stop(not start_btn.value, mo.md("⏸ Chọn dataset, nhập token rồi bấm ▶."))
    _tok = hf_token_ui.value.strip()
    mo.stop(not _tok, mo.md("⚠ Chưa nhập HF token: nhập, nhấn Enter, rồi bấm ▶ lại."))
    _task, _ck = TASKS[dataset_ui.value]
    _runs = f"/marimo/runs/cmr_{_ck}"
    os.makedirs(_runs + "/logs", exist_ok=True)
    _pidf = _runs + "/run_cmr.pid"
    if os.path.exists(_pidf) and os.path.exists("/proc/" + open(_pidf).read().strip()):
        _msg = f"CMR {dataset_ui.value} đang chạy; xem tiến độ bên dưới."
    else:
        _p = subprocess.Popen(
            ["bash", os.path.join(REPO_ROOT, "scripts", "cmr_molab_run.sh")],
            env={**os.environ, "HF_TOKEN": _tok, "REPO_ROOT": REPO_ROOT, "CMR_TASK": _task},
            start_new_session=True, stdout=open(_runs + "/logs/run_cmr.out", "a"), stderr=subprocess.STDOUT,
        )
        open(_pidf, "w").write(str(_p.pid))
        _msg = f"Đã khởi chạy CMR {dataset_ui.value} (pid {_p.pid})."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["1m", "5m"], default_interval="1m", label="Làm mới tiến độ")
    refresh
    return (refresh,)


@app.cell
def _(TASKS, dataset_ui, glob, json, mo, os, refresh):
    # Tiến độ của dataset đang chọn: các bước của script, valid theo epoch (độ chính xác trong batch của CMR),
    # sao lưu, metric test
    refresh
    _task, _ck = TASKS[dataset_ui.value]
    _logs = f"/marimo/runs/cmr_{_ck}/logs"
    _steps = open(f"{_logs}/progress.txt").read() if os.path.exists(f"{_logs}/progress.txt") else "(chưa chạy)"
    _rows = []
    _runs = sorted(glob.glob(f"/marimo/baselines/CMR/ckpt/{_ck}/{_task.lower()}-*/log.txt"), key=os.path.getmtime)
    if _runs:
        for _l in open(_runs[-1], encoding="utf-8", errors="replace"):
            if "valid metric" in _l:
                _ep = _l.split("Epoch ")[1].split(",")[0]
                _m = json.loads(_l.split("valid metric: ")[1])
                _rows.append(f"| {_ep} | {_m['Acc@1']} | {_m['MMAcc@1']} | {_m['loss']} |")
    _backup = open(f"{_logs}/backup.log").read().splitlines()[-3:] if os.path.exists(f"{_logs}/backup.log") else []
    _metrics = ""
    for _v in ("noknn", "knn"):
        _f = f"/marimo/runs/cmr_{_ck}/eval/metrics_cmr_{_v}_test.md"
        if os.path.exists(_f):
            _metrics += f"\n**Test, {_v}:**\n\n" + "\n".join(open(_f).read().splitlines()[:8]) + "\n"
    mo.md(
        f"### {dataset_ui.value}: các bước\n```\n" + _steps + "\n```\n"
        + ("### Valid theo epoch (Acc@1 = ảnh, dùng để dừng sớm; MMAcc@1 = text + ảnh)\n"
           "| Epoch | Acc@1 | MMAcc@1 | loss |\n|---|---|---|---|\n" + "\n".join(_rows) + "\n" if _rows else "")
        + ("### Sao lưu\n```\n" + "\n".join(_backup) + "\n```\n" if _backup else "")
        + _metrics
    )
    return


if __name__ == "__main__":
    app.run()
