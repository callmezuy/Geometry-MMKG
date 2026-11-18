"""
Read the raw Spatial-VRD annotations (sg_dataset JSON) as triples with the raw predicate.

Entity id = "<photo_id>_<obj_idx>" (the ids of data/Spatial-VRD/entity2id.txt). Used by
scripts/build_vrd_relation_review.py (label proposals) and scripts/clean_splits.py (building the splits).
"""
import json
import os


def rebuild(json_path, entity2id):
    """[(head, raw_predicate, tail, example_text)] for every relationship whose two objects are in entity2id."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [{"filename": k, **v} for k, v in data.items()]
    out = []
    for img in data:
        fn = img.get("filename")
        pid = str(img.get("photo_id", os.path.splitext(fn)[0]))
        for rel in img.get("relationships", []):
            objs = rel.get("objects")
            if not objs or len(objs) != 2:
                continue
            s, o = f"{pid}_{objs[0]}", f"{pid}_{objs[1]}"
            if s in entity2id and o in entity2id:
                out.append((s, rel.get("relationship", "unknown"), o, " ".join(rel.get("text", [])[1:2])))
    return out
