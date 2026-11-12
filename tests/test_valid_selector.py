"""
Check baselines/{NativE,SNAG}/.../mmkgc/config/ValidSelector.py (validation-based checkpoint selection).
Run from Code/:  python tests/test_valid_selector.py
"""
import filecmp
import importlib.util
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = [os.path.join(ROOT, "baselines", "NativE", "mmkgc", "config", "ValidSelector.py"),
         os.path.join(ROOT, "baselines", "SNAG", "SNAG_MKGC", "mmkgc", "config", "ValidSelector.py")]


def load(path):  # import the file alone (the mmkgc packages load the OpenKE C++ library)
    spec = importlib.util.spec_from_file_location("ValidSelector", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ToyKGE(nn.Module):
    """OpenKE-style model: predict() returns scores where LOWER is better."""
    def __init__(self, n, r, d=6):
        super().__init__()
        self.ent_tot = n
        self.ent = nn.Embedding(n, d)
        self.rel = nn.Embedding(r, d)
        self.frozen = nn.Embedding.from_pretrained(torch.randn(n, d)).requires_grad_(False)

    def predict(self, data):
        h, t, r = self.ent(data["batch_h"]), self.ent(data["batch_t"]), self.rel(data["batch_r"])
        return (h + r - t).abs().sum(-1).detach().cpu().numpy()


def write_split(d, name, triples):
    with open(os.path.join(d, name), "w") as f:
        f.write(f"{len(triples)}\n")
        for h, r, t in triples:
            f.write(f"{h} {t} {r}\n")  # OpenKE order: h t r


def test_valid_selector():
    assert filecmp.cmp(PATHS[0], PATHS[1], shallow=False), "NativE and SNAG copies differ"
    vs = load(PATHS[0])
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, R = 120, 3
    trip = sorted({(int(rng.integers(N)), int(rng.integers(R)), int(rng.integers(N))) for _ in range(600)})
    trip += [(4, 1, int(t)) for t in range(10, 20)] + [(int(h), 2, 9) for h in range(30, 40)]
    trip = sorted(set(trip))
    train, valid, test = trip[0::3], trip[1::3], trip[2::3]
    with tempfile.TemporaryDirectory() as d:
        for n, s in (("train2id.txt", train), ("valid2id.txt", valid), ("test2id.txt", test)):
            write_split(d, n, s)
        model = ToyKGE(N, R)
        logs = []
        sel = vs.ValidSelector(d, every=5, patience=2, log=logs.append)

        # 1) ranks equal an independent brute force (strict ties, filtered, head + tail)
        E, Rm = model.ent.weight.detach(), model.rel.weight.detach()
        tails, heads = {}, {}
        for h, r, t in trip:
            tails.setdefault((h, r), set()).add(t); heads.setdefault((r, t), set()).add(h)
        ranks = []
        for h, r, t in valid:
            sh = (E + Rm[r] - E[t]).abs().sum(1)      # every entity as head
            st = (E[h] + Rm[r] - E).abs().sum(1)      # every entity as tail
            ranks.append(1 + sum(1 for e in range(N) if sh[e] < sh[h] and e not in heads[(r, t)]))
            ranks.append(1 + sum(1 for e in range(N) if st[e] < st[t] and e not in tails[(h, r)]))
        exp = np.mean(1 / np.array(ranks, dtype=float))
        got = sel.evaluate(model)["mrr"]
        assert abs(got - exp) < 1e-12, (got, exp)

        # 2) best weights kept, restored; frozen tables are not copied
        assert sel.due(4) and not sel.due(3)
        assert not sel.step(model, 4)
        best = model.ent.weight.detach().clone()
        assert "frozen.weight" not in sel.best_state
        with torch.no_grad():
            model.ent.weight.add_(torch.randn_like(model.ent.weight) * 5)
        worse = sel.best_mrr / 2   # random weights are not reliably worse; control the metric
        sel.evaluate = lambda m: {"mrr": worse, "mr": 0.0, "hit10": 0.0, "hit3": 0.0, "hit1": 0.0}
        assert not sel.step(model, 9)          # 1 evaluation without improvement
        assert sel.step(model, 14)             # 2 -> patience reached -> stop
        sel.restore_best(model)
        assert torch.equal(model.ent.weight.detach(), best) and sel.best_epoch == 4

        # 3) seeded subset
        sub = vs.ValidSelector(d, max_triples=20, seed=1, log=lambda *_: None)
        sub2 = vs.ValidSelector(d, max_triples=20, seed=1, log=lambda *_: None)
        assert len(sub.triples) == 20 and sub.triples == sub2.triples and set(sub.triples) <= set(valid)
    print(f"OK: {len(valid)} valid triples, MRR {got:.4f} == brute force; best epoch restored; patience works")


if __name__ == "__main__":
    test_valid_selector()
