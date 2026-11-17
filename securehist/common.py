"""Shared helpers: dataset listing, statistics, CNN loading and attacks."""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                     # .../project 1
DATA_DIR = os.path.join(PROJECT, "lung_colon_image_set")
CLASSES = ["colon_aca", "colon_n", "lung_aca"]      # order used when the CNN was trained
CNN_INPUT = 180
SEED = 42
DEV_PER_CLASS = 10   # first 10 images of each class folder were used to fix the watermark thresholds


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def list_images(split: str = "Test Set", limit_per_class: int | None = None, classes=CLASSES,
                exclude_dev: bool = False):
    """[(path, label_idx, class_name)] sorted, from <DATA_DIR>/<split>/<class>/*.jpeg.
    exclude_dev drops the first DEV_PER_CLASS images of each folder (used to fix thresholds)."""
    out = []
    for idx, c in enumerate(classes):
        files = sorted(glob.glob(os.path.join(DATA_DIR, split, c, "*.jpeg")))
        if exclude_dev:
            files = files[DEV_PER_CLASS:]
        if limit_per_class:
            files = files[:limit_per_class]
        out += [(f, idx, c) for f in files]
    if not out:
        sys.exit(f"No images found under {os.path.join(DATA_DIR, split)}")
    return out


def out_dir(name: str) -> str:
    d = os.path.join(PROJECT, "results_v2", name)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96):
    """Proportion with 95% Wilson interval -> (p, lo, hi) as fractions."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{100*p:.2f} [{100*lo:.2f}, {100*hi:.2f}] (n={n})"


def bootstrap_ci(values, groups=None, n_boot: int = 2000, stat=np.mean, seed: int = SEED):
    """Mean with a 95% percentile bootstrap CI; resamples groups (e.g. images) if given."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = [np.array([i]) for i in range(len(values))]
    else:
        groups = np.asarray(groups)
        idx = [np.where(groups == g)[0] for g in np.unique(groups)]
    est = float(stat(values))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        boots.append(stat(values[np.concatenate([idx[i] for i in pick])]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar test on paired booleans -> (b_only, a_only, p_value)."""
    from scipy.stats import binomtest
    a = np.asarray(a_correct, bool)
    b = np.asarray(b_correct, bool)
    n01 = int((~a & b).sum())   # b right, a wrong
    n10 = int((a & ~b).sum())   # a right, b wrong
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(n01, n, 0.5).pvalue
    return n01, n10, float(p)


# ------------------------------------------------------------------- CNN + attacks
def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn(weights: str):
    """CNNModel from medical_classifier/model.py with the given weights file."""
    import torch
    sys.path.insert(0, PROJECT)
    from medical_classifier.model import CNNModel
    path = weights if os.path.isabs(weights) else os.path.join(PROJECT, weights)
    m = CNNModel(num_classes=len(CLASSES))
    m.load_state_dict(torch.load(path, map_location=device(), weights_only=True))
    return m.to(device()).eval()


def to_cnn_input(x):
    """Full-resolution batch in [0,1] (B,3,H,W) -> CNN input (B,3,180,180), differentiable."""
    import torch.nn.functional as F
    if x.shape[-1] == CNN_INPUT and x.shape[-2] == CNN_INPUT:
        return x
    return F.interpolate(x, size=(CNN_INPUT, CNN_INPUT), mode="bilinear", align_corners=False, antialias=True)


def predict(model, x):
    """Returns (pred, confidence, probs) for a batch already in [0,1]."""
    import torch
    with torch.no_grad():
        p = torch.softmax(model(to_cnn_input(x)), dim=1)
    conf, pred = p.max(1)
    return pred, conf, p


def fgsm(model, x, y, eps, mask=None):
    import torch
    import torch.nn.functional as F
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(to_cnn_input(x)), y)
    g = torch.autograd.grad(loss, x)[0].sign()
    if mask is not None:
        g = g * mask
    return torch.clamp(x + eps * g, 0, 1).detach()


def pgd(model, x, y, eps, alpha, iters, mask=None, random_start=True, project_fn=None):
    """Untargeted L_inf PGD. mask (B,1,H,W) restricts the perturbation to a region.
    project_fn(adv) is an optional extra projection applied after each step (adaptive)."""
    import torch
    import torch.nn.functional as F
    x0 = x.clone().detach()
    adv = x0.clone()
    if random_start:
        noise = torch.empty_like(x0).uniform_(-eps, eps)
        adv = torch.clamp(x0 + (noise * mask if mask is not None else noise), 0, 1)
    for _ in range(iters):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(to_cnn_input(adv)), y)
        g = torch.autograd.grad(loss, adv)[0].sign()
        if mask is not None:
            g = g * mask
        adv = adv.detach() + alpha * g
        adv = torch.clamp(x0 + torch.clamp(adv - x0, -eps, eps), 0, 1)
        if project_fn is not None:
            adv = project_fn(adv)
    return adv.detach()


def to_tensor_batch(rgb_list):
    import torch
    arr = np.stack(rgb_list).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device())


def to_uint8_list(x):
    arr = (x.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return list(arr)


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        n = time.time()
        d, self.t = n - self.t, n
        return d
