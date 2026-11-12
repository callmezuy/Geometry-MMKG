# coding:utf-8
"""
[Geometry-MMKG] Checkpoint selection on the VALIDATION split (RESEARCH_PLAN.md, P3.0).

Upstream behaviour differs between baselines:
  * SNAG keeps the checkpoint with the best TEST MRR and early-stops on it (test leakage);
  * NativE trains a fixed number of epochs and tests the last checkpoint.
For the comparison every model selects its checkpoint on validation MRR and touches the
test split exactly once. Use `-select_on valid` (run scripts); `-select_on original` keeps the
upstream behaviour (needed to check that our copy still reproduces the published numbers).

Evaluation is pure Python (no OpenKE C++ state, so it cannot disturb the test loader):
filtered ranks over ALL entities, head + tail pooled, no type constraint, strict tie rule
(rank = 1 + #entities with a strictly lower score, other true answers from train+valid+test
excluded) -- the same protocol as Tester.run_link_prediction and core/models/evaluation.py.

This file is identical in baselines/NativE/mmkgc/config/ and baselines/SNAG/SNAG_MKGC/mmkgc/config/.
"""
import os

import numpy as np
import torch


def read_triples(path):
    """OpenKE *2id.txt: first line = count, then 'h t r' per line. Returns list of (h, r, t)."""
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            x = line.split()
            if len(x) >= 3:
                out.append((int(x[0]), int(x[2]), int(x[1])))
    return out


def load_filters(in_path):
    tails, heads = {}, {}
    for fn in ("train2id.txt", "valid2id.txt", "test2id.txt"):
        p = os.path.join(in_path, fn)
        if os.path.exists(p):
            for h, r, t in read_triples(p):
                tails.setdefault((h, r), set()).add(t)
                heads.setdefault((r, t), set()).add(h)
    return tails, heads


def filtered_rank(score, true_id, true_set):
    """score: lower is better (OpenKE predict)."""
    s = np.asarray(score).reshape(-1)
    v = s[true_id]
    better = int((s < v).sum())
    others = [e for e in true_set if e != true_id]
    if others:
        better -= int((s[np.asarray(others, dtype=np.int64)] < v).sum())
    return better + 1


class ValidSelector(object):
    """Evaluate every `every` epochs on the validation split, keep the best weights in memory.

    Trainers call  stop = selector.step(kge_model, epoch)  and  selector.restore_best(kge_model).
    patience > 0: stop after `patience` evaluations without improvement (0 = never stop early,
    i.e. the same epoch budget as upstream). max_triples > 0: evaluate on a fixed random subset
    (seeded) of the validation split, for very large graphs (Spatial-VG).
    """

    def __init__(self, in_path, split="valid2id.txt", every=10, patience=0, max_triples=0, seed=0, log=print):
        self.every = max(1, int(every))
        self.patience = int(patience)
        self.log = log
        triples = read_triples(os.path.join(in_path, split))
        if not triples:
            raise ValueError(f"ValidSelector: no triples in {os.path.join(in_path, split)}")
        if max_triples and len(triples) > max_triples:
            idx = np.sort(np.random.default_rng(seed).choice(len(triples), size=max_triples, replace=False))
            triples = [triples[i] for i in idx]
        self.triples = triples
        self.tails, self.heads = load_filters(in_path)
        self.best_mrr, self.best_epoch, self.best_state = -1.0, -1, None
        self.bad, self.history = 0, []
        self.log(f"[ValidSelector] {split}: {len(triples)} triples, every {self.every} epochs, "
                 f"patience {self.patience or 'off'}")

    def due(self, epoch):
        return (epoch + 1) % self.every == 0

    @torch.no_grad()
    def evaluate(self, model):
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        all_e = torch.arange(model.ent_tot, dtype=torch.long, device=dev)
        ranks = []
        for h, r, t in self.triples:
            H = torch.tensor([h], dtype=torch.long, device=dev)
            R = torch.tensor([r], dtype=torch.long, device=dev)
            T = torch.tensor([t], dtype=torch.long, device=dev)
            s = model.predict({"batch_h": all_e, "batch_t": T, "batch_r": R, "mode": "head_batch"})
            ranks.append(filtered_rank(s, h, self.heads.get((r, t), ())))
            s = model.predict({"batch_h": H, "batch_t": all_e, "batch_r": R, "mode": "tail_batch"})
            ranks.append(filtered_rank(s, t, self.tails.get((h, r), ())))
        if was_training:
            model.train()
        rk = np.asarray(ranks, dtype=np.float64)
        return {"mrr": float(np.mean(1 / rk)), "mr": float(rk.mean()), "hit10": float(np.mean(rk <= 10)),
                "hit3": float(np.mean(rk <= 3)), "hit1": float(np.mean(rk <= 1))}

    def step(self, model, epoch):
        """Evaluate, remember the best weights. Returns True when training should stop."""
        m = self.evaluate(model)
        self.history.append((epoch, m["mrr"]))
        if m["mrr"] > self.best_mrr:
            self.best_mrr, self.best_epoch, self.bad = m["mrr"], epoch, 0
            # Only weights that change: trainable parameters + buffers (frozen modality
            # tables can be GBs and never change).
            st = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
            st.update({k: b.detach().cpu().clone() for k, b in model.named_buffers()})
            self.best_state = st
        else:
            self.bad += 1
        self.log(f"[ValidSelector] epoch {epoch}: valid MRR {m['mrr']:.4f} | MR {m['mr']:.1f} | "
                 f"H@1 {m['hit1']:.4f} | H@10 {m['hit10']:.4f}  (best {self.best_mrr:.4f} @ epoch {self.best_epoch})")
        return self.patience > 0 and self.bad >= self.patience

    def restore_best(self, model):
        if self.best_state is None:
            self.log("[ValidSelector] no validation run happened; keeping the last weights")
            return
        missing = [k for k in self.best_state if k not in dict(model.named_parameters()) and k not in dict(model.named_buffers())]
        if missing:
            raise RuntimeError(f"[ValidSelector] unknown keys in saved state: {missing[:5]}")
        model.load_state_dict(self.best_state, strict=False)
        self.log(f"[ValidSelector] restored best weights: epoch {self.best_epoch}, valid MRR {self.best_mrr:.4f}")
