# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True, other_model=None, norm=False, mu=0.5):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.other_model = other_model
        self.norm = norm
        self.mu = mu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
            
            

    def run_link_prediction(self, type_constrain=False, rank_file=None):
        """rank_file: if given, also compute filtered head/tail ranks per test triple in Python
        (same rule as the C++ tester: strictly lower score is better, other true triples from
        train/valid/test filtered, no type constraint) and write them as CSV
        h_id,r_id,t_id,rank_head,rank_tail for Geometry-MMKG's scripts/compute_metrics.py."""
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        if rank_file:
            tails, heads = self._load_filters()
            dumped = []
        training_range = self.data_loader
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            if rank_file:
                h, t, r = int(data_tail["batch_h"][0]), int(data_head["batch_t"][0]), int(data_head["batch_r"][0])
                rank_head = self._filtered_rank(score, h, heads.get((r, t), ()))
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
            if rank_file:
                rank_tail = self._filtered_rank(score, t, tails.get((h, r), ()))
                dumped.append((h, r, t, rank_head, rank_tail))
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        if rank_file:
            self._write_ranks(rank_file, dumped, (mrr, mr, hit10, hit3, hit1))
        return mrr, mr, hit10, hit3, hit1

    def _load_filters(self):
        """All true triples (train+valid+test, OpenKE files 'h t r') as {(h,r):{t}} and {(r,t):{h}}."""
        tails, heads = {}, {}
        for fn in ("train2id.txt", "valid2id.txt", "test2id.txt"):
            path = os.path.join(self.data_loader.in_path, fn)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                next(f)
                for line in f:
                    x = line.split()
                    if len(x) < 3:
                        continue
                    h, t, r = int(x[0]), int(x[1]), int(x[2])
                    tails.setdefault((h, r), set()).add(t)
                    heads.setdefault((r, t), set()).add(h)
        return tails, heads

    @staticmethod
    def _filtered_rank(score, true_id, true_set):
        s = np.asarray(score).reshape(-1)
        v = s[true_id]
        better = int((s < v).sum())
        others = [e for e in true_set if e != true_id]
        if others:
            better -= int((s[np.asarray(others, dtype=np.int64)] < v).sum())
        return better + 1

    @staticmethod
    def _write_ranks(path, rows, cpp_metrics):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["h_id", "r_id", "t_id", "rank_head", "rank_tail"])
            w.writerows(rows)
        ranks = np.array([[r[3], r[4]] for r in rows], dtype=np.float64).reshape(-1)
        py = (np.mean(1 / ranks), np.mean(ranks), np.mean(ranks <= 10), np.mean(ranks <= 3), np.mean(ranks <= 1))
        print(f"Saved {len(rows)} per-triple ranks to {path}")
        print("Rank check  (MRR, MR, H@10, H@3, H@1)  C++: " + ", ".join(f"{x:.6f}" for x in cpp_metrics)
              + "  | Python: " + ", ".join(f"{x:.6f}" for x in py))

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod
