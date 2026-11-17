"""How much do false alarms depend on the secret key?

The watermark bits come from the key, so which untouched images trip the localization rule changes with the key
(first run: 13/1,470 flagged, second key: 33/1,470). This script repeats the untouched-image test with K
independent keys and, for each key, reports:
  * false alarms on untouched watermarked images at several tamper thresholds (0.19 published ... 0.23)
  * detection of a 64x64 copy-move edit (one random site per image) at the same thresholds
  * the threshold the development procedure would pick for that key (smallest with 0/80 development images flagged)
The keys are fixed public strings ("key-variation-0", ...): a false-alarm rate does not depend on keeping a key
secret, only on its bits, so the experiment is reproducible.

Usage:  python securehist/eval_key_variation.py [--keys 5] [--per-class N] [--limit 3]
Output: results_v2/key_variation/{cases.csv, dev.csv, summary.csv}
"""
import argparse
import os
import zlib
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd

import common as C
import wm_core as W
from eval_embed_ablation import dev_images, load
from eval_system import PID
from eval_watermark import site_boxes, t_splice

THRESHOLDS = [0.19, 0.20, 0.21, 0.22, 0.23]
GRID = np.round(np.arange(0.15, 0.301, 0.01), 2)


def flags_at(img, cfg, key):
    """Image-level 'not intact' decision at every threshold from a single distance computation."""
    dist, ok = W.block_distances(img, PID, cfg, key)
    nc = float(2.0 * ok.mean() - 1.0)
    score = W.uniform_filter(dist, cfg.smoothing, mode="nearest")
    out = {}
    for t in sorted(set(THRESHOLDS) | set(GRID)):
        f = score > t
        f = W.binary_opening(f, structure=np.ones((2, 2), dtype=bool))
        if f.any():
            lab, n = W.label(f)
            sizes = np.bincount(lab.ravel())
            f = (sizes[1:] >= cfg.min_region_blocks).any()
        else:
            f = False
        out[float(t)] = bool(nc < cfg.nc_intact or f)
    return nc, out


def summarize(out):
    df = pd.read_csv(os.path.join(out, "cases.csv"))
    dev = pd.read_csv(os.path.join(out, "dev.csv")) if os.path.exists(os.path.join(out, "dev.csv")) else None
    rows = []
    for key, g in df.groupby("key"):
        r = {"key": key, "n_images": len(g), "min_nc": float(g.nc.min())}
        if dev is not None and (dev.key == key).any():
            d = dev[dev.key == key]
            free = [t for t in GRID if not d[f"fa_{t:.2f}"].any()]
            r["dev_threshold"] = float(free[0]) if free else np.nan
        for t in THRESHOLDS:
            r[f"fa_{t:.2f}"] = 100 * g[f"fa_{t:.2f}"].mean()
            r[f"det64_{t:.2f}"] = 100 * g[f"det64_{t:.2f}"].mean()
        rows.append(r)
    s = pd.DataFrame(rows)
    agg = {"key": "all keys (pooled)", "n_images": int(len(df)), "min_nc": float(df.nc.min())}
    if "dev_threshold" in s:
        agg["dev_threshold"] = float(s.dev_threshold.max())
    for t in THRESHOLDS:
        p, lo, hi = C.wilson(int(df[f"fa_{t:.2f}"].sum()), len(df))
        agg[f"fa_{t:.2f}"], agg[f"fa_{t:.2f}_lo"], agg[f"fa_{t:.2f}_hi"] = 100 * p, 100 * lo, 100 * hi
        agg[f"fa_{t:.2f}_keymin"], agg[f"fa_{t:.2f}_keymax"] = float(s[f"fa_{t:.2f}"].min()), float(s[f"fa_{t:.2f}"].max())
        agg[f"det64_{t:.2f}"] = 100 * df[f"det64_{t:.2f}"].mean()
    s = pd.concat([s, pd.DataFrame([agg])], ignore_index=True)
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    cols = ["key", "dev_threshold"] + [f"fa_{t:.2f}" for t in THRESHOLDS] + [f"det64_{t:.2f}" for t in THRESHOLDS]
    print(s[[c for c in cols if c in s]].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=None, help="subsample test images per class (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = C.out_dir("key_variation")
    if args.summarize_only:
        summarize(out); return
    cfg = W.WMConfig()
    keys = [f"key-variation-{i}" for i in range(args.keys)]
    items = C.list_images("Test Set", args.limit or args.per_class, exclude_dev=True)
    devs = dev_images()
    csv_path, dev_path = os.path.join(out, "cases.csv"), os.path.join(out, "dev.csv")
    rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []
    drows = pd.read_csv(dev_path).to_dict("records") if os.path.exists(dev_path) else []
    done = {(r["key"], r["image"]) for r in rows}
    done_dev = {r["key"] for r in drows}
    for key in keys:
        kb = key.encode()
        if key not in done_dev:
            for p in devs:
                nc, fl = flags_at(W.embed(load(p), PID, cfg, kb), cfg, kb)
                drows.append({"key": key, "image": os.path.basename(p), **{f"fa_{t:.2f}": fl[float(t)] for t in GRID}})
            pd.DataFrame(drows).to_csv(dev_path, index=False)
        for n, (path, label, cname) in enumerate(items, 1):
            img_id = os.path.relpath(path, C.DATA_DIR)
            if (key, img_id) in done:
                continue
            ws = W.embed(load(path), PID, cfg, kb)
            nc, fl = flags_at(ws, cfg, kb)
            rng = np.random.default_rng(zlib.crc32((key + img_id).encode("utf-8")))
            box = site_boxes(rng, ws.shape[0], ws.shape[1], 64, 1)[0]
            _, fl64 = flags_at(t_splice(ws, box, None), cfg, kb)
            rows.append({"key": key, "image": img_id, "cls": cname, "nc": nc,
                         **{f"fa_{t:.2f}": fl[t] for t in THRESHOLDS}, **{f"det64_{t:.2f}": fl64[t] for t in THRESHOLDS}})
            if n % 50 == 0 or n == len(items):
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"  {key}: {n}/{len(items)}")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summarize(out)


if __name__ == "__main__":
    main()
