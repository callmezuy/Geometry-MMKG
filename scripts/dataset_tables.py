"""Bảng thống kê kiểu paper cho 6 bộ dữ liệu v0 (Markdown + LaTeX booktabs).

Chạy từ Code/:
    python scripts/dataset_tables.py                                   # in Markdown
    python scripts/dataset_tables.py --out experiment_results/dataset_tables.md \
        --latex experiment_results/dataset_tables.tex

Cần: data/<name>/ (split, *2id, text, image_index.tsv, ent_img_mask.pt), Spatial-VRD/sg_*_annotations.json,
Spatial-VG/VG-SGG.h5 + VG-SGG-dicts.json, data/raw/vg/image_data.json.zip (kích thước ảnh VG, để tính box).
Spatial-VG không có ent_img_mask.pt ở máy (chỉ trên HF): mặt nạ được tính lại bằng đúng luật của
scripts/extract_features.py (box nằm hẳn ngoài ảnh = thiếu ảnh).

Bảng:
  T1 Tổng quan (như Table 1 của MMRNS / NativE / CMR): #Ent, #Rel, #Train/#Valid/#Test, #Images.
  T2 Modality: độ phủ ảnh, ảnh/thực thể, ảnh giữ chỗ đã loại, văn bản (độ dài, bị cắt 512 token),
     test chia theo số đầu bộ ba có ảnh.
  T3 Cấu trúc đồ thị: bậc, thực thể bậc 1, thực thể cô lập, loại ánh xạ 1-1/1-N/N-1/N-N (Bordes 2013), độ lệch quan hệ.
  T4 Split: thực thể theo split, tỉ lệ chưa thấy, test chia theo đầu nào là thực thể mới (như BLP / GraIL).
  T5 Scene graph (như VG150 / Neural Motifs / BGNN): ảnh, lớp vật thể, box, box/ảnh, bộ ba/ảnh,
     predicate head/body/tail (ngưỡng BGNN: >10k / 0,5k-10k / <0,5k bộ ba train), diện tích box, box chủ-khách chồng nhau.
  T6 Nhóm quan hệ GEOMETRIC / POSSESSIVE / SEMANTIC / MISC (định nghĩa của Zellers et al. 2018).
  T7 Đối chiếu với số công bố của nguồn gốc.
"""
import argparse
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_stats import DATASETS, SPLITS, load_dataset, mapping_types, read_count_file  # noqa: E402

GRAPH = {"FB15k-237-IMG": "MMKG", "FB15K-237-ind": "MMKG", "MKG-Y": "MMKG",
         "Spatial-VRD": "MMSG", "Spatial-VRD-Trans": "MMSG", "Spatial-VG": "MMSG"}
SETTING = {"FB15k-237-IMG": "Trans.", "FB15K-237-ind": "Ind.", "MKG-Y": "Trans.",
           "Spatial-VRD": "Ind.", "Spatial-VRD-Trans": "Trans.", "Spatial-VG": "Ind."}
CATS = ["GEOMETRIC", "POSSESSIVE", "SEMANTIC", "MISC"]
VG_CORRUPT = {1592, 1722, 4616, 4617}  # dropped by VG-SGG.h5 (same as extract_features.py)


# --------------------------------------------------------------------------- tables
class Table:
    def __init__(self, key, title, header, align, note=""):
        self.key, self.title, self.header, self.align, self.note = key, title, header, align, note
        self.rows = []

    def add(self, *cells):
        assert len(cells) == len(self.header), (self.key, cells)
        self.rows.append([str(c) for c in cells])

    def md(self):
        sep = ["---:" if a == "r" else "---" for a in self.align]
        L = [f"### {self.key}. {self.title}", "", "| " + " | ".join(self.header) + " |", "|" + "|".join(sep) + "|"]
        L += ["| " + " | ".join(r) + " |" for r in self.rows]
        if self.note:
            L += ["", self.note]
        return "\n".join(L + [""])

    def tex(self):
        esc = lambda s: (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
                         .replace("#", "\\#").replace("_", "\\_").replace("~", "$\\sim$")
                         .replace("≥", "$\\geq$").replace("<", "$<$").replace(">", "$>$").replace("×", "$\\times$")
                         .replace("✓", "$\\checkmark$").replace("✗", "$\\times$").replace("≈", "$\\approx$")
                         .replace("**", "").replace("`", ""))
        L = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{esc(self.title)}}}",
             f"\\label{{tab:data-{self.key.lower()}}}", "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{" + "".join(self.align) + "}", "\\toprule",
             " & ".join(esc(h) for h in self.header) + " \\\\", "\\midrule"]
        L += [" & ".join(esc(c) for c in r) + " \\\\" for r in self.rows]
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
        return "\n".join(L)


def n(x):
    return f"{x:,}"


def p(a, b, d=1):
    return f"{100.0 * a / b:.{d}f}" if b else "–"


# --------------------------------------------------------------------------- per-dataset facts
def read_text(path, has_count):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        if has_count:
            f.readline()
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            out[k] = v
    return out


def load_mask(d):
    f = os.path.join(d, "ent_img_mask.pt")
    if not os.path.exists(f):
        return None
    import torch

    return torch.load(f, map_location="cpu", weights_only=True).numpy().astype(bool)


def zip_sizes(zip_paths, wanted):
    """{file name: (W, H)} read from the image headers in the zips (the size extract_features.py sees)."""
    from PIL import Image

    out = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                b = m.rsplit("/", 1)[-1]
                if b in wanted and b not in out:
                    with z.open(m) as f:
                        out[b] = Image.open(f).size
    return out


def vrd_scene(d, ents, zip_path):
    """VRD: entity -> (image, object class, box xyxy, image W, H) from sg_*_annotations.json.

    Spatial-VRD-Trans has the same entity2id and reads the JSON of Spatial-VRD."""
    if not os.path.exists(os.path.join(d, "sg_train_annotations.json")):
        d = os.path.join(os.path.dirname(d), "Spatial-VRD")
    idx = {e: i for i, e in enumerate(ents)}
    img = np.full(len(ents), -1, np.int64)
    cls = [None] * len(ents)
    box = np.zeros((len(ents), 4))
    wh = np.zeros((len(ents), 2))
    images, meta = [], []
    for js in ["sg_train_annotations.json", "sg_test_annotations.json"]:
        with open(os.path.join(d, js), encoding="utf-8") as f:
            data = json.load(f)
        for x in data:
            pid = str(x.get("photo_id", os.path.splitext(x["filename"])[0]))
            ii = len(images)
            images.append(pid)
            meta.append((x["filename"], x["width"], x["height"]))
            for k, o in enumerate(x.get("objects", [])):
                e = idx.get(f"{pid}_{k}")
                if e is None:
                    continue
                b = o["bbox"]
                img[e], cls[e] = ii, o["names"][0].strip().lower()
                box[e] = (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
    assert (img >= 0).all()
    real = zip_sizes([zip_path], {m[0] for m in meta})
    sizes = np.array([real.get(fn, (w, h)) for fn, w, h in meta], np.float64)
    wh[:] = sizes[img]
    json_diff = sum(1 for fn, w, h in meta if fn in real and real[fn] != (w, h))
    return img, cls, box, wh, {"json_size_differs": json_diff, "not_in_zip": len(meta) - len(real)}


def vg_scene(d, image_data_zip, zip_paths):
    import h5py

    with h5py.File(os.path.join(d, "VG-SGG.h5"), "r") as h:
        boxes = h["boxes_1024"][:].astype(np.float64)
        labels = h["labels"][:, 0]
        fb, lb = h["img_to_first_box"][:], h["img_to_last_box"][:]
    with open(os.path.join(d, "VG-SGG-dicts.json"), encoding="utf-8") as f:
        i2l = json.load(f)["idx_to_label"]
    with zipfile.ZipFile(image_data_zip) as z:
        im = json.loads(z.read([m for m in z.namelist() if m.endswith(".json")][0]))
    im = [x for x in im if x["image_id"] not in VG_CORRUPT]
    assert len(im) == len(fb)
    real = zip_sizes(zip_paths, {f"{x['image_id']}.jpg" for x in im})
    N = len(boxes)
    img = np.full(N, -1, np.int64)
    wh = np.zeros((N, 2))
    box = np.zeros((N, 4))
    for i, meta in enumerate(im):
        if fb[i] < 0:
            continue
        sl = slice(fb[i], lb[i] + 1)
        s = max(meta["width"], meta["height"]) / 1024.0
        bb = boxes[sl] * s
        x1, y1 = bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2
        box[sl] = np.stack([x1, y1, x1 + bb[:, 2], y1 + bb[:, 3]], 1)
        # boxes are scaled with the metadata size (as in extract_features.py), then checked against the real file
        img[sl], wh[sl] = i, real.get(f"{meta['image_id']}.jpg", (meta["width"], meta["height"]))
    assert (img >= 0).all()
    cls = [i2l[str(c)] for c in labels.tolist()]
    diff = sum(1 for x in im if f"{x['image_id']}.jpg" in real and real[f"{x['image_id']}.jpg"] != (x["width"], x["height"]))
    return img, cls, box, wh, int((fb >= 0).sum()), len(fb), {"json_size_differs": diff, "not_in_zip": len(im) - len(real)}


def outside(box, wh):
    return (box[:, 0] >= wh[:, 0]) | (box[:, 1] >= wh[:, 1]) | (box[:, 2] <= 0) | (box[:, 3] <= 0)


def dataset_facts(name, d, raw_root):
    ents, rels, trip = load_dataset(d)
    N, R = len(ents), len(rels)
    F = {"name": name, "N": N, "R": R, "rels": rels, "trip": trip, "n": {s: len(trip[s]) for s in SPLITS}}
    tr = trip["train"]
    seen = set(tr[:, 0].tolist()) | set(tr[:, 2].tolist())
    F["split_ents"] = {s: set(trip[s][:, 0].tolist()) | set(trip[s][:, 2].tolist()) for s in SPLITS}
    used = set().union(*F["split_ents"].values())
    F["isolated"] = N - len(used)
    # degree on train
    deg = np.bincount(np.concatenate([tr[:, 0], tr[:, 2]]), minlength=N)[sorted(seen)]
    F["deg"] = deg
    # split / unseen
    te = trip["test"]
    hs = np.isin(te[:, 0], list(seen))
    ts = np.isin(te[:, 2], list(seen))
    F["test_seen"] = {"both": int((hs & ts).sum()), "head_new": int((~hs & ts).sum()),
                      "tail_new": int((hs & ~ts).sum()), "both_new": int((~hs & ~ts).sum())}
    F["unseen"] = {s: len(F["split_ents"][s] - seen) for s in SPLITS}
    # BLP / CMR view: "new" = test entity, i.e. not in train or valid triples
    tv = list(seen | F["split_ents"]["valid"])
    hs2, ts2 = np.isin(te[:, 0], tv), np.isin(te[:, 2], tv)
    F["test_seen_tv"] = {"head_new": int((~hs2 & ts2).sum()), "tail_new": int((hs2 & ~ts2).sum()),
                         "both_new": int((~hs2 & ~ts2).sum())}
    # relations
    rc = Counter(tr[:, 1].tolist())
    F["rel_train"] = rc
    F["mapping"] = mapping_types(tr, R)
    cat_file = os.path.join(d, "relation_category.tsv")
    if os.path.exists(cat_file):
        with open(cat_file, encoding="utf-8") as f:
            next(f)
            F["cat"] = {int(q[1]): q[2] for q in (l.rstrip("\n").split("\t") for l in f) if len(q) >= 3}
    # modalities
    fs = os.path.join(d, "feature_stats.json")
    F["feat"] = json.load(open(fs, encoding="utf-8")) if os.path.exists(fs) else {}
    cs = os.path.join(d, "clean_stats.json")
    F["clean"] = json.load(open(cs, encoding="utf-8")) if os.path.exists(cs) else {}
    short = read_text(os.path.join(d, "entity2text.txt"), name == "Spatial-VG")
    long_ = read_text(os.path.join(d, "entity2textlong.txt"), False)
    texts = [long_.get(e) or short.get(e, "") for e in ents]
    F["text_words"] = np.array([len(t.split()) for t in texts])
    F["has_desc"] = sum(1 for e in ents if e in long_ and long_[e] != short.get(e)) if long_ else None
    F["text_unique"] = len(set(texts))
    mask = load_mask(d)
    if GRAPH[name] == "MMKG":
        per = Counter()
        status = Counter()
        with open(os.path.join(d, "image_index.tsv"), encoding="utf-8") as f:
            for line in f:
                e, _, st = line.rstrip("\n").split("\t")
                status[st] += 1
                if st == "ok":
                    per[e] += 1
        F["img_status"] = status
        F["img_per_ent"] = np.array([per[e] for e in ents if per[e] > 0])
        F["n_images"] = status["ok"]
        if mask is None:
            mask = np.array([per[e] > 0 for e in ents])
    else:
        if name.startswith("Spatial-VRD"):
            img, cls, box, wh, F["size_check"] = vrd_scene(
                d, ents, os.path.join(raw_root, "vrd", "visual-relationship-detection-vrd-dataset.zip"))
            F["n_images"] = len(np.unique(img))
        else:
            img, cls, box, wh, F["n_images"], F["n_h5_images"], F["size_check"] = vg_scene(
                d, os.path.join(raw_root, "vg", "image_data.json.zip"),
                [os.path.join(raw_root, "vg", z) for z in ("images.zip", "images2.zip")])
        out = outside(box, wh)
        if mask is None:
            mask = ~out
        F["boxes_outside"] = int(out.sum())
        if mask is not None and not np.array_equal(mask, ~out):
            print(f"  [{name}] ent_img_mask.pt != recomputed mask: {int((mask != ~out).sum())} rows", file=sys.stderr)
        print(f"  [{name}] image sizes: {F['size_check']}, boxes outside: {F['boxes_outside']}", file=sys.stderr)
        F["img"], F["cls"], F["box"], F["wh"] = img, cls, box, wh
    F["mask"] = mask
    assert len(mask) == N
    return F


# --------------------------------------------------------------------------- build tables
def build(facts):
    T = []
    # ---- T1 overview
    t = Table("T1", "Tổng quan các bộ dữ liệu", ["Dataset", "Graph", "Setting", "#Ent", "#Rel", "#Train", "#Valid",
                                               "#Test", "#Images"], "lllrrrrrr",
              "#Ent = toàn bộ `entity2id` (= không gian ứng viên khi xếp hạng, mục 4.2 của RESEARCH_PLAN). "
              "#Images: MMKG = số ảnh dùng được (trạng thái `ok` trong `image_index.tsv`); "
              "MMSG = số ảnh cảnh có ít nhất một box. Trans. = transductive, Ind. = inductive.")
    for F in facts:
        t.add(F["name"], GRAPH[F["name"]], SETTING[F["name"]], n(F["N"]), F["R"], *(n(F["n"][s]) for s in SPLITS),
              n(F["n_images"]))
    T.append(t)

    # ---- T2 modalities
    t = Table("T2", "Độ phủ và đặc điểm các modality",
              ["Dataset", "Ent. w/ image (%)", "Img/ent. mean / med. / max", "Images removed",
               "Ent. w/ description (%)", "Text words mean / med. / max", "Truncated @512 tok.",
               "Test: both / one / no image"], "lrrrrrrr",
              "MMKG: một thực thể có nhiều ảnh (đặc trưng = trung bình); MMSG: mỗi thực thể là một box (1 crop). "
              "Images removed = ảnh đọc được nhưng bị loại khỏi tập `ok`: ảnh giữ chỗ, gần một màu, ảnh trùng trong cùng "
              "thực thể (P0.1, P0.3; không tính ảnh không đọc được: FB15k-237 0, MKG-Y 3). MMSG: box nằm hẳn ngoài "
              "ảnh (lỗi chú thích) được tính là thiếu ảnh. "
              "Văn bản: `entity2textlong.txt` nếu có, ngược lại `entity2text.txt` (VRD: thuộc tính + tên vật thể; "
              "VG: tên lớp). \"Ent. w/ description\" = có mô tả dài khác tên. Cột cuối: số bộ ba test có cả hai / một / "
              "không đầu nào có ảnh (mục 4.10).")
    for F in facts:
        m = F["mask"]
        te = F["trip"]["test"]
        k = m[te[:, 0]].astype(int) + m[te[:, 2]].astype(int)
        if GRAPH[F["name"]] == "MMKG":
            ipe = F["img_per_ent"]
            ipe_s = f"{ipe.mean():.2f} / {np.median(ipe):.0f} / {ipe.max()}"
            rem = sum(v for s, v in F["img_status"].items() if s != "ok")
            rem_s = f"{n(rem)} ({p(rem, rem + F['img_status']['ok'], 2)}%)"
        else:
            ipe_s, rem_s = "1 / 1 / 1", f"{F['boxes_outside']} box ngoài ảnh"
        tw = F["text_words"]
        trunc = F["feat"].get("encoders", {}).get("text_stats", {}).get("truncated_strings")
        if trunc is None and tw.max() < 100:
            trunc = 0
        desc = f"{p(F['has_desc'], F['N'], 2)}" if F["has_desc"] is not None else "–"
        t.add(F["name"], f"{n(int(m.sum()))} ({p(m.sum(), F['N'], 2)})", ipe_s, rem_s, desc,
              f"{tw.mean():.1f} / {np.median(tw):.0f} / {n(int(tw.max()))}", trunc,
              f"{n(int((k == 2).sum()))} / {n(int((k == 1).sum()))} / {n(int((k == 0).sum()))}")
    T.append(t)

    # ---- T3 graph structure
    t = Table("T3", "Cấu trúc đồ thị (tính trên train)",
              ["Dataset", "Deg. mean / med. / max", "Deg.-1 ent. (%)", "Isolated ent.", "Triples / ent.",
               "1-1 / 1-N / N-1 / N-N (#rel)", "Test in N-N (%)", "Top-5 rel. (% train)", "Rel. < 10 train"],
              "lrrrrrrrr",
              "Bậc = số bộ ba train chứa thực thể (đầu hoặc đuôi), chỉ tính thực thể có trong train. "
              "Isolated = thực thể có trong `entity2id` nhưng không thuộc bộ ba nào (MMSG: box không có quan hệ; "
              "vẫn nằm trong không gian ứng viên). Loại ánh xạ theo Bordes et al. (2013), ngưỡng 1,5 trên train. "
              "Ở MMSG mỗi thực thể là một box riêng nên gần như mọi quan hệ thành 1-1: đây là tính chất của đồ thị "
              "cảnh (thực thể không lặp lại giữa các ảnh), không phải lỗi.")
    for F in facts:
        deg = F["deg"]
        mt = Counter(F["mapping"].values())
        te = F["trip"]["test"]
        nn = sum(1 for r in te[:, 1].tolist() if F["mapping"].get(r) == "N-N")
        freq = sorted(F["rel_train"].values(), reverse=True)
        ntr = F["n"]["train"]
        t.add(F["name"], f"{deg.mean():.2f} / {np.median(deg):.0f} / {n(int(deg.max()))}",
              p((deg == 1).sum(), len(deg)), n(F["isolated"]), f"{ntr / len(deg):.2f}",
              " / ".join(str(mt.get(x, 0)) for x in ["1-1", "1-N", "N-1", "N-N"]), p(nn, len(te)),
              p(sum(freq[:5]), ntr), sum(1 for c in freq if c < 10))
    T.append(t)

    # ---- T4 split / inductive
    t = Table("T4", "Thực thể theo split và mức độ inductive",
              ["Dataset", "Ent. train / valid / test", "Unseen ent. valid (%)", "Unseen ent. test (%)",
               "Test: both seen", "Test: only head new", "Test: only tail new", "Test: both new"], "lrrrrrrr",
              "Thực thể theo split = thực thể xuất hiện trong bộ ba của split đó. \"New\" = không xuất hiện trong "
              "bộ ba train. Cách chia test theo đầu nào là thực thể mới giống Daza et al. (BLP, 2021).")
    ind = [F for F in facts if F["name"] == "FB15K-237-ind"]
    if ind:
        q = ind[0]["test_seen_tv"]
        t.note += (f" FB15K-237-ind: test có cả thực thể của valid (cũng chưa thấy khi train). Nếu chỉ coi thực thể "
                   f"của test (không có trong train ∪ valid) là mới, như `test_entities.txt`: chỉ đầu mới "
                   f"{n(q['head_new'])}, chỉ đuôi mới {n(q['tail_new'])}, cả hai mới {n(q['both_new'])}.")
    for F in facts:
        se, un, ts = F["split_ents"], F["unseen"], F["test_seen"]
        nt = F["n"]["test"]
        t.add(F["name"], " / ".join(n(len(se[s])) for s in SPLITS), p(un["valid"], len(se["valid"])),
              p(un["test"], len(se["test"])),
              *(f"{n(ts[k])} ({p(ts[k], nt)}%)" for k in ["both", "head_new", "tail_new", "both_new"]))
    T.append(t)

    # ---- T5 scene graphs
    sg = [F for F in facts if GRAPH[F["name"]] == "MMSG"]
    t = Table("T5", "Thống kê đồ thị cảnh (MMSG)",
              ["Dataset", "Images train / valid / test", "#Obj. classes (≥ 10 boxes)", "#Pred.", "#Boxes (in ≥1 triple)",
               "Boxes / img.", "Triples / img.", "Pred. head / body / tail", "Box area / img. (med.)",
               "Subj.-obj. boxes overlap (%)"], "lrrrrrrrrr",
              "Ảnh theo split = ảnh có ít nhất một bộ ba trong split (Spatial-VRD-Trans chia theo bộ ba nên một ảnh "
              "nằm ở nhiều split). #Obj. classes: VRD là tên vật thể tự do (`names[0]`, chữ thường), VG là 150 lớp. "
              "Boxes / img. tính mọi box của ảnh; Triples / img. tính trên ảnh có bộ ba. Head / body / tail theo "
              "ngưỡng của BGNN (Li et al., 2021): > 10.000 / 500–10.000 / < 500 bộ ba train. Diện tích box = phần box "
              "nằm trong ảnh chia diện tích ảnh. Overlap = tỉ lệ bộ ba (mọi split) có box chủ thể và khách thể giao nhau.")
    for F in sg:
        img, box, wh = F["img"], F["box"], F["wh"]
        imgs = {}
        for s in SPLITS:
            x = F["trip"][s]
            imgs[s] = len(np.unique(img[np.concatenate([x[:, 0], x[:, 2]])]))
        allt = np.concatenate([F["trip"][s] for s in SPLITS])
        in_trip = np.zeros(F["N"], bool)
        in_trip[allt[:, 0]] = in_trip[allt[:, 2]] = True
        n_img = F["n_images"]
        tpi = len(allt) / len(np.unique(img[allt[:, 0]]))
        rc = F["rel_train"]
        hbt = (sum(1 for c in rc.values() if c > 10000), sum(1 for c in rc.values() if 500 <= c <= 10000),
               sum(1 for c in rc.values() if c < 500))
        cb = np.stack([np.clip(box[:, 0], 0, wh[:, 0]), np.clip(box[:, 1], 0, wh[:, 1]),
                       np.clip(box[:, 2], 0, wh[:, 0]), np.clip(box[:, 3], 0, wh[:, 1])], 1)
        area = (cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1]) / (wh[:, 0] * wh[:, 1])
        a, b = box[allt[:, 0]], box[allt[:, 2]]
        iw = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
        ih = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
        ov = ((iw > 0) & (ih > 0)).mean()
        cc = Counter(F["cls"])
        t.add(F["name"], " / ".join(n(imgs[s]) for s in SPLITS),
              f"{n(len(cc))} ({n(sum(1 for v in cc.values() if v >= 10))})", F["R"],
              f"{n(F['N'])} ({n(int(in_trip.sum()))})", f"{F['N'] / n_img:.2f}", f"{tpi:.2f}",
              " / ".join(map(str, hbt)), f"{100 * np.median(area):.1f}%", f"{100 * ov:.1f}")
    T.append(t)

    # ---- T6 relation categories
    t = Table("T6", "Nhóm quan hệ (phân loại của chúng tôi theo định nghĩa của Zellers et al., 2018)",
              ["Dataset"] + [f"{c} #rel / train % / test %" for c in CATS], "lrrrr",
              "Bảng gán từng quan hệ vào nhóm: `relation_category.tsv` của mỗi bộ MMSG (phụ lục).")
    for F in sg:
        cat = F["cat"]
        tr, te = F["trip"]["train"], F["trip"]["test"]
        crel = Counter(cat.values())
        ctr = Counter(cat.get(r, "?") for r in tr[:, 1].tolist())
        cte = Counter(cat.get(r, "?") for r in te[:, 1].tolist())
        t.add(F["name"], *(f"{crel.get(c, 0)} / {p(ctr[c], len(tr))} / {p(cte[c], len(te))}" for c in CATS))
    T.append(t)

    # ---- T7 check against published numbers
    by = {F["name"]: F for F in facts}
    t = Table("T7", "Đối chiếu với số công bố của nguồn gốc",
              ["Dataset", "Source", "Published", "Ours (v0)", "Match"], "lllll",
              "Chỉ liệt kê số mà nguồn gốc công bố. Ảnh VRD thiếu là `4392556686_44d71ff5a0_o.gif` (ảnh test duy nhất "
              "dạng GIF; 23 object, 23 relationship trong JSON): không có thực thể nào của ảnh này trong `entity2id`. "
              "Spatial-VRD dùng annotation tự do `sg_dataset` (không phải bản "
              "100 lớp / 70 predicate của Lu et al., 2016) và gán lại predicate (P0.4), nên chỉ so được số ảnh. "
              "Spatial-VG dùng VG150-curated của Neau et al. (2023), cùng tập 108.073 ảnh và cùng từ vựng "
              "150 lớp / 50 predicate với VG150 (Xu et al., 2017).")

    def row(name, src, pub, ours):
        t.add(name, src, pub, ours, "✓" if pub == ours else "✗")

    if "FB15k-237-IMG" in by:
        F = by["FB15k-237-IMG"]
        row("FB15k-237-IMG", "Toutanova & Chen (2015); split KG-BERT / MKGformer",
            "14,541 / 237 / 272,115 / 17,535 / 20,466", f"{n(F['N'])} / {F['R']} / " + " / ".join(n(F['n'][s]) for s in SPLITS))
    if "FB15K-237-ind" in by:
        F = by["FB15K-237-ind"]
        se = F["split_ents"]
        row("FB15K-237-ind", "Daza et al. (2021); CMR Table 1",
            "ent. 11,633 / 1,454 / 1,454; triples 215,082 / 42,164 / 52,870",
            f"ent. {n(len(se['train']))} / {n(len(se['valid'] - se['train']))} / "
            f"{n(len(se['test'] - se['train'] - se['valid']))}; triples " + " / ".join(n(F['n'][s]) for s in SPLITS))
    if "MKG-Y" in by:
        F = by["MKG-Y"]
        row("MKG-Y", "Xu et al. (2022, MMRNS); NativE", "15,000 / 28 / 21,310 / 2,665 / 2,663",
            f"{n(F['N'])} / {F['R']} / " + " / ".join(n(F['n'][s]) for s in SPLITS))
    if "Spatial-VRD" in by:
        F = by["Spatial-VRD"]
        t.add("Spatial-VRD", "Lu et al. (2016), `sg_dataset`", "5,000 images (4,000 train / 1,000 test)",
              f"{n(F['n_images'])} images ({n(len(np.unique(F['img'][F['trip']['test'][:, 0]])))} test)",
              "≈ (thiếu 1 ảnh test GIF)")
    if "Spatial-VG" in by:
        F = by["Spatial-VG"]
        row("Spatial-VG", "Xu et al. (2017); Neau et al. (2023)", "108,073 images / 150 classes / 50 predicates",
            f"{n(F['n_h5_images'])} images / {len(set(F['cls']))} classes / {F['R']} predicates")
    T.append(t)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--latex", default=None)
    a = ap.parse_args()
    facts = []
    for name in a.datasets:
        print(f"[{name}] ...", file=sys.stderr)
        facts.append(dataset_facts(name, os.path.join(a.data_root, name), os.path.join(a.data_root, "raw")))
    T = build(facts)
    md = "\n".join(["# Thống kê dữ liệu (bảng kiểu paper)", "",
                    "Sinh bởi `scripts/dataset_tables.py` từ `data/` (v0). Thống kê kiểm tra chi tiết từng bộ: "
                    "`dataset_stats.md`.", ""] + [x.md() for x in T])
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
    if a.latex:
        with open(a.latex, "w", encoding="utf-8") as f:
            f.write("% Sinh bởi scripts/dataset_tables.py. Cần \\usepackage{booktabs,graphicx,amssymb}.\n\n")
            f.write("\n".join(x.tex() for x in T))


if __name__ == "__main__":
    main()
