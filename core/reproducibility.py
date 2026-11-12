"""Seeding helpers (RESEARCH_PLAN.md P1.2). Default seed 42 = NativE's default (-seed 42)."""
import os
import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=False):
    """Seed python, numpy and torch (CPU + CUDA). Returns a torch.Generator for DataLoader(shuffle=True).

    deterministic=True also forces deterministic cuDNN kernels (slower; cdist/scatter ops may
    still be non-deterministic on GPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    """DataLoader(worker_init_fn=seed_worker): give every worker a reproducible seed."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
