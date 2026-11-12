"""
Propose a clean relation label for every RAW Spatial-VRD predicate (sg_dataset JSON)
and write a review sheet for manual checking.

Target vocabulary: the 50 predicates of VG150-curated "no_part_whole"
(Neau et al., ICCVW 2023), i.e. exactly the vocabulary of Spatial-VG, plus a few
frequent VRD predicates missing from it (VRD_EXTRA: "left of", "right of", ...).
Rules work on WORDS (never substrings) and are applied in priority order:
  passive inversion > explicit phrase > direction word > posture+preposition
  > proximity > verb family > bare preposition > drop.

Output columns (data/Spatial-VRD/relation_review.tsv):
  raw_relationship   predicate as written in the JSON "relationship" field
  count              number of triples (entities present in entity2id)
  normalized         after lower-casing, typo fixes, removing is/are/articles
  proposed_label     target label, or <DROP>
  inverse            1 = swap head and tail (passive voice, e.g. "worn by")
  category           GEOMETRIC / POSSESSIVE / SEMANTIC / MISC of the label
  rule               which rule produced the label
  needs_review       non-empty = please check this row (reason given)
  old_label          label produced by the old clean_mmsg_aliases.py
  examples           raw "text" predicates from the JSON

Edit proposed_label / inverse / category directly in the TSV; clean_splits.py
will read the edited file.

Usage (from Code/):
  python scripts/build_vrd_relation_review.py
"""
import argparse
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Vocabulary and categories
# Category scheme: the Geometric / Possessive / Semantic / Misc TYPES of Neural Motifs
# (Zellers et al., CVPR 2018, Sec. 3, Table 1). Checked 11/11/2026 against the paper:
#   * The paper gives only 3 examples per type plus class counts and instance shares
#     (Geometric 15 cls / 50.0%: above, behind, under; Possessive 8 / 40.9%: has,
#     part of, wearing; Semantic 24 / 8.7%: carrying, eating, using; Misc 3 / 0.3%:
#     for, from, made of). It does NOT publish the full 50-predicate assignment, so
#     this table is OUR assignment following those type definitions; cite it as such.
#   * Consistent with all paper examples; Misc in Motifs is exactly {for, from, made of}.
#   * Known deviations, measured on VG150 predicate_count: "and"/"to" kept in MISC
#     (Motifs' counts imply they sit in Geometric/Semantic; together <0.3% of triples);
#     our Possessive share is 44.0% vs 40.9%, Geometric 47.9% vs 50.0%: the paper's
#     list must differ by ~3% of instances, but the data cannot pin down which predicate.
#   * Neau et al. 2023 (VG150-curated) reuse these types and give no per-predicate list.
# The same CATEGORY table is used for Spatial-VG (see clean_splits.py).
# ---------------------------------------------------------------------------
CATEGORY = {
    # geometric
    "above": "GEOMETRIC", "across": "GEOMETRIC", "against": "GEOMETRIC", "along": "GEOMETRIC",
    "at": "GEOMETRIC", "attached to": "GEOMETRIC", "behind": "GEOMETRIC", "between": "GEOMETRIC",
    "in front of": "GEOMETRIC", "near": "GEOMETRIC", "on": "GEOMETRIC", "outside": "GEOMETRIC",
    "over": "GEOMETRIC", "under": "GEOMETRIC", "underneath": "GEOMETRIC",
    # possessive
    "has": "POSSESSIVE", "in": "POSSESSIVE", "of": "POSSESSIVE", "wearing": "POSSESSIVE",
    "with": "POSSESSIVE", "worn by": "POSSESSIVE",
    # semantic (actions / interactions; many carry a spatial component)
    "carrying": "SEMANTIC", "covered in": "SEMANTIC", "covering": "SEMANTIC", "crossing": "SEMANTIC",
    "eating": "SEMANTIC", "flying": "SEMANTIC", "flying in": "SEMANTIC", "hanging from": "SEMANTIC",
    "holding": "SEMANTIC", "laying on": "SEMANTIC", "looking at": "SEMANTIC", "lying on": "SEMANTIC",
    "mounted on": "SEMANTIC", "painted on": "SEMANTIC", "parked on": "SEMANTIC", "playing": "SEMANTIC",
    "riding": "SEMANTIC", "says": "SEMANTIC", "sitting at": "SEMANTIC", "sitting on": "SEMANTIC",
    "standing on": "SEMANTIC", "using": "SEMANTIC", "walking in": "SEMANTIC", "walking on": "SEMANTIC",
    "watching": "SEMANTIC",
    # misc
    "and": "MISC", "for": "MISC", "from": "MISC", "to": "MISC",
}
assert len(CATEGORY) == 50
# VRD extras: frequent in VRD, absent from the curated vocabulary. Kept so spatial information is not lost.
VRD_EXTRA = {
    "left of": "GEOMETRIC", "right of": "GEOMETRIC", "touching": "GEOMETRIC", "around": "GEOMETRIC",
    "taller than": "GEOMETRIC", "larger than": "GEOMETRIC", "facing": "GEOMETRIC",
    "pulling": "SEMANTIC", "cutting": "SEMANTIC",
}
CATEGORY.update(VRD_EXTRA)
VOCAB = set(CATEGORY)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
TYPO = {
    "behing": "behind", "behnid": "behind", "behinds": "behind", "bhind": "behind", "ehind": "behind",
    "behindt": "behind", "aboe": "above", "abovw": "above", "benesth": "beneath", "beneatht": "beneath",
    "belowc": "below", "undernearth": "underneath", "infront": "in front", "ontop": "on top",
    "nex": "next", "tp": "to", "carring": "carrying", "wesrs": "wears", "ridings": "riding",
    "suports": "supports", "attatched": "attached", "atached": "attached", "hearding": "herding",
    "wearnig": "wearing", "nera": "near", "nextto": "next to", "onb": "on",
}
STOP = {"is", "are", "was", "were", "be", "being", "a", "an", "the", "its", "his", "her", "their", "some"}


def normalize(raw):
    s = re.sub(r"[^a-z ]+", " ", raw.lower())
    toks = [TYPO.get(t, t) for t in s.split()]
    toks = " ".join(toks).split()  # typo fixes may introduce spaces
    toks = [t for t in toks if t not in STOP]
    return " ".join(toks)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
PHRASE = {
    "next to": "near", "beside": "near", "adjacent to": "near", "close to": "near", "alongside": "near",
    "near": "near", "by": "near", "next": "near",
    "below": "under", "beneath": "under", "underneath": "underneath", "under": "under",
    "above": "above", "over": "over",
    "on top of": "on", "atop": "on", "upon": "on", "on": "on", "onto": "on",
    "inside": "in", "inside of": "in", "within": "in", "in": "in", "into": "in",
    "has": "has", "have": "has", "having": "has", "contains": "has", "contain": "has", "containing": "has",
    "with": "with", "of": "of", "for": "for", "from": "from", "to": "to", "and": "and", "at": "at",
    "wearing": "wearing", "wear": "wearing", "wears": "wearing", "has on": "wearing", "dressed in": "wearing",
    "holding": "holding", "holds": "holding", "hold": "holding",
    "carrying": "carrying", "carries": "carrying", "carry": "carrying",
    # part-whole predicates are excluded in VG150-curated "no_part_whole"
    "made of": None, "part of": None, "belongs to": None, "belonging to": None, "worn by": "worn by",
    "covered in": "covered in", "covered with": "covered in",
    "on back of": "on", "on side of": "on",
    "across": "across", "across from": "across", "against": "against", "along": "along", "between": "between",
    "says": "says", "say": "says", "reads": "says",
    # fixes for phrases where the last preposition is misleading
    "ahead of": "in front of", "before": "in front of", "in back of": "behind", "back of": "behind",
    "beyond": "behind", "in between": "between", "in middle of": "between",
    "outside": "outside", "outside of": "outside", "outside on": "outside",
    "far from": None, "away from": None, "talking to": None, "listening to": None, "waiting for": None,
    "talking on": "using", "on phone": "using",
    # VRD extras
    "touching": "touching", "touch": "touching", "touches": "touching",
    "around": "around", "surrounding": "around", "surround": "around", "surrounds": "around",
    "wrapped around": "around", "encircling": "around",
    "taller than": "taller than", "higher than": "taller than", "larger than": "larger than",
    "bigger than": "larger than",
    "facing": "facing", "faces": "facing", "face": "facing",
    "pulling": "pulling", "pulls": "pulling", "pull": "pulling", "towing": "pulling",
    "crossing": "crossing", "crosses": "crossing", "cross": "crossing",
    "cutting": "cutting", "cuts": "cutting", "cut": "cutting",
}
# inverse phrases: (label, inverse=1)
PHRASE_INV = {
    "shorter than": "taller than", "lower than": "taller than",
    "smaller than": "larger than", "surrounded by": "around", "pulled by": "pulling",
    "supported by": None, "held up by": "holding",
}
# phrases whose mapping is a judgement call -> always flagged
PHRASE_REVIEW = {
    "by": "by merged into near?", "contains": "contains merged into has?", "contain": "contains merged into has?",
    "across from": "across from merged into across?", "wears": "wears merged into wearing (as in VG150-curated)",
    "reads": "reads -> says?", "on side of": "on side of -> on or near?", "over": "over kept separate from above (VG150)",
    "before": "before = in front of (spatial) or temporal?", "beyond": "beyond -> behind?",
    "in middle of": "in middle of -> between?", "facing": "facing: geometric orientation, VRD extra",
}

# passive "<participle> by" -> (active label, inverse=1)
PASSIVE = {
    "held": "holding", "carried": "carrying", "covered": "covering",
    "ridden": "riding", "used": "using", "eaten": "eating", "watched": "watching", "owned": "has",
}

DIRECTION = [  # (word, label); "front" handled as "in front of"
    ("left", "left of"), ("right", "right of"), ("behind", "behind"), ("front", "in front of"),
    ("under", "under"), ("underneath", "underneath"), ("below", "under"), ("beneath", "under"),
    ("above", "above"), ("over", "over"),
]

POSTURE = {  # first word family -> {preposition: label}
    ("sitting", "sits", "sit", "seated", "perched"): {"on": "sitting on", "in": "sitting on", "at": "sitting at"},
    ("standing", "stands", "stand"): {"on": "standing on", "in": "standing on"},
    ("lying", "lies", "lie"): {"on": "lying on", "in": "lying on"},
    ("laying", "lays", "lay"): {"on": "laying on", "in": "laying on"},
    ("parked", "parks", "park", "parking"): {"on": "parked on", "in": "parked on", "at": "parked on"},
    ("walking", "walks", "walk"): {"on": "walking on", "in": "walking in", "across": "walking on"},
    ("mounted", "mounts"): {"on": "mounted on", "to": "mounted on"},
    ("painted", "printed", "drawn", "written", "engraved"): {"on": "painted on"},
    ("growing", "grows", "grow"): {"on": "on", "in": "in", "from": "on"},
    ("hanging", "hangs", "hang", "hung"): {"on": "hanging from", "from": "hanging from", "off": "hanging from", "in": "hanging from"},
    ("flying", "flies", "fly"): {"in": "flying in", "over": "flying in", "above": "flying in"},
}

VERB = {  # first word -> label (verb used without a decisive preposition)
    "holding": "holding", "holds": "holding", "hold": "holding", "grabbing": "holding", "grasping": "holding",
    "grasps": "holding", "grabs": "holding", "gripping": "holding",
    "carrying": "carrying", "carries": "carrying", "carry": "carrying",
    "wearing": "wearing", "wears": "wearing", "wear": "wearing",
    "eating": "eating", "eats": "eating", "eat": "eating", "biting": "eating", "grazing": "eating",
    "drinking": "eating", "feeding": "eating", "chewing": "eating",
    "looking": "looking at", "looks": "looking at", "staring": "looking at", "facing": "looking at",
    "watching": "watching", "watches": "watching", "watch": "watching",
    "riding": "riding", "rides": "riding", "ride": "riding", "driving": "riding", "drives": "riding",
    "using": "using", "uses": "using", "use": "using", "operating": "using",
    "playing": "playing", "plays": "playing", "play": "playing", "swinging": "playing", "throwing": "playing",
    "hitting": "playing", "kicking": "playing", "catching": "playing",
    "covering": "covering", "covers": "covering", "cover": "covering",
    "attached": "attached to", "connected": "attached to", "tied": "attached to", "plugged": "attached to",
    "fastened": "attached to", "chained": "attached to", "hooked": "attached to", "mounted": "mounted on",
    "says": "says", "reads": "says", "showing": "says", "shows": "says", "displaying": "says",
    "flying": "flying", "flies": "flying", "fly": "flying",
}
VERB.update({"touching": "touching", "pulling": "pulling", "crossing": "crossing",
             "cutting": "cutting", "facing": "facing"})
VERB_REVIEW = {"drinking", "swinging", "throwing", "hitting", "kicking", "catching", "pushing",
               "operating", "driving", "drives", "showing", "shows", "displaying"}

# bare-preposition fallback, checked in this order; maps the word to a label
PREP_FALLBACK = [("on", "on"), ("onto", "on"), ("inside", "in"), ("into", "in"), ("within", "in"),
                 ("in", "in"), ("at", "at"), ("across", "across"), ("against", "against"),
                 ("along", "along"), ("between", "between"), ("around", "around"), ("outside", "outside"),
                 ("with", "with")]
# NOTE: "of", "from", "to", "for" are NOT used as fallbacks ("ahead of" -> of would be wrong).


def propose(norm):
    """Return (label or None, inverse, rule, review_reason)."""
    toks = norm.split()
    if not toks:
        return None, 0, "empty", ""
    # 1. explicit phrases (incl. inverse ones)
    if norm in PHRASE_INV:
        lab = PHRASE_INV[norm]
        return lab, (1 if lab else 0), "phrase-inverse", ""
    if norm in PHRASE:
        return PHRASE[norm], 0, "phrase", PHRASE_REVIEW.get(norm, "")
    # 2. passive voice "<participle> by"; "<verb-ing> by" means proximity ("standing by")
    m = re.match(r"^(\w+)(?: up)? by$", norm)
    if m:
        w = m.group(1)
        if w in PASSIVE:
            return PASSIVE[w], 1, "passive", ""
        if w.endswith("ing"):
            return "near", 0, "proximity", ""
        return None, 0, "passive-unknown", "passive voice with no active label in vocab"
    # 3. direction word anywhere (word-level)
    for w, lab in DIRECTION:
        if w in toks:
            if w == "front" and not ("in" in toks or "of" in toks):
                continue
            return lab, 0, "direction", ""
    # 4. posture/verb + preposition
    first = toks[0]
    for fam, preps in POSTURE.items():
        if first in fam:
            for p in toks[1:]:
                if p in preps:
                    return preps[p], 0, "posture+prep", ""
    # 5. proximity
    if any(t in toks for t in ("near", "beside", "adjacent", "alongside", "next")) or "close to" in norm:
        return "near", 0, "proximity", ""
    # 6. verb family (first word)
    if first in VERB:
        return VERB[first], 0, "verb", ("verb family is a judgement call" if first in VERB_REVIEW else "")
    if first in PASSIVE and len(toks) > 1 and toks[1] == "by":
        return PASSIVE[first], 1, "passive", ""
    # 7. bare preposition fallback, e.g. "floating on" -> on
    for p, lab in PREP_FALLBACK:
        if p in toks:
            return lab, 0, "prep-fallback", ""
    return None, 0, "unmapped", ""


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Spatial-VRD")
    ap.add_argument("--out", default=None)
    ap.add_argument("--review-min-count", type=int, default=5,
                    help="flag dropped / fallback rows with at least this many triples")
    args = ap.parse_args()
    out = args.out or os.path.join(args.data, "relation_review.tsv")
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    spec = importlib.util.spec_from_file_location("audit", os.path.join(code_dir, "scripts", "audit_vrd_relations.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    old_map = audit.load_old_mapper(code_dir)

    entity2id = {}
    with open(os.path.join(args.data, "entity2id.txt"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            entity2id[p[0]] = int(p[1])
    raw = audit.rebuild(os.path.join(args.data, "sg_train_annotations.json"), entity2id) + \
          audit.rebuild(os.path.join(args.data, "sg_test_annotations.json"), entity2id)

    cnt = Counter(p for _, p, _, _ in raw)
    examples = defaultdict(list)
    for _, p, _, txt in raw:
        if txt and txt not in examples[p] and len(examples[p]) < 3:
            examples[p].append(txt)

    rows, label_mass, review_mass, drop_mass = [], Counter(), 0, 0
    for p, n in cnt.most_common():
        norm = normalize(p)
        lab, inv, rule, reason = propose(norm)
        if lab is not None and lab not in VOCAB:
            reason = reason or f"label '{lab}' not in vocabulary"
        if lab is None and n >= args.review_min_count and not rule.startswith("phrase"):
            reason = reason or "dropped although frequent"
        if rule == "prep-fallback" and n >= 2 * args.review_min_count:
            reason = reason or "verb ignored, only preposition kept"
        if rule == "direction" and n >= args.review_min_count and norm != lab and len(norm.split()) > 2:
            reason = reason or "long phrase reduced to its direction word"
        if lab in VRD_EXTRA and lab not in ("left of", "right of") and n >= 2 * args.review_min_count:
            reason = reason or "VRD extra label (not in VG150)"
        rows.append((p, n, norm, lab or "<DROP>", inv, CATEGORY.get(lab, "") if lab else "",
                     rule, reason, old_map(p), " | ".join(examples[p])))
        if lab:
            label_mass[lab] += n
        else:
            drop_mass += n
        if reason:
            review_mass += n

    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("raw_relationship\tcount\tnormalized\tproposed_label\tinverse\tcategory\trule\tneeds_review\told_label\texamples\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

    total = sum(cnt.values())
    n_rev = sum(1 for r in rows if r[7])
    print(f"raw predicates: {len(cnt)} | triples: {total}")
    print(f"labels used: {len(label_mass)} / {len(VOCAB)} in vocabulary")
    print(f"dropped: {sum(1 for r in rows if r[3] == '<DROP>')} predicates, {drop_mass} triples ({100 * drop_mass / total:.2f}%)")
    print(f"needs review: {n_rev} predicates, {review_mass} triples ({100 * review_mass / total:.2f}%)")
    print(f"inverted (passive): {sum(r[1] for r in rows if r[4])} triples")
    changed = sum(r[1] for r in rows if r[3] != r[8] and r[3] != '<DROP>')
    print(f"label differs from old heuristic: {changed} triples")
    by_cat = Counter()
    for lab, n in label_mass.items():
        by_cat[CATEGORY[lab]] += n
    print("per category:", dict(by_cat.most_common()))
    print("\nlabel counts:")
    for lab, n in label_mass.most_common():
        print(f"  {lab:14} {n:6}  {CATEGORY[lab]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
