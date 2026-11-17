"""Experiment B: sensitivity to the perturbation budget.

300 test images (100 per class, fixed random sample, development images excluded), watermarked.
For eps in {0, 0.25, 0.5, 1, 2, 4, 8}/255: FGSM and PGD-20 (step eps/4, random start) against the deployed,
defended CNN at full resolution, rounded to 8-bit. eps = 0 is the untouched control.

Per case: any pixel changed after rounding, PSNR to the watermarked image, watermark rule flag, pixel-hash
flag (= changed), CNN-confidence flag and harmful success (correct on the clean image, wrong after attack).
CNN-confidence flag: confidence below the 5th percentile of the clean (eps = 0) confidences of these images,
i.e. a 5% false-alarm operating point, as in Table 4.

Usage:  python securehist/eval_eps_sweep.py [--per-class 100]   |   --summarize-only
Output: results_v2/eps_sweep/{cases.csv, summary.csv}
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

import common as C
import wm_core as W
from eval_system import PID, cnn_signals

EPS = [0, 0.25, 0.5, 1, 2, 4, 8]
ATTACKS = ["fgsm", "pgd20"]


def summarize(out):
    df = pd.read_csv(os.path.join(out, "cases.csv"))
    clean = df[(df.eps == 0) & (df.attack == "fgsm")]
    tau = float(np.quantile(clean.cnn_conf, 0.05))
    df["flag_cnn"] = df.cnn_conf < tau
    rows = []
    for eps in EPS:
        r = {"eps_255": eps}
        g_all = df[df.eps == eps]
        r["changed"] = 100 * g_all.changed.mean()
        ch = g_all[g_all.changed]
        r["psnr_db"] = float(ch.psnr.mean()) if len(ch) else np.nan
        for a in ATTACKS:
            g = df[(df.eps == eps) & (df.attack == a)]
            ok = g[g.clean_correct]
            for key, col in (("wm", "flag_wm"), ("hash", "changed"), ("cnn", "flag_cnn")):
                p, lo, hi = C.wilson(int(g[col].sum()), len(g))
                r[f"{key}_{a}"] = 100 * p; r[f"{key}_{a}_lo"] = 100 * lo; r[f"{key}_{a}_hi"] = 100 * hi
            p, lo, hi = C.wilson(int(ok.wrong_after.sum()), len(ok))
            r[f"harmful_{a}"] = 100 * p; r[f"harmful_{a}_lo"] = 100 * lo; r[f"harmful_{a}_hi"] = 100 * hi
            r[f"n_{a}"] = len(g)
        rows.append(r)
    s = pd.DataFrame(rows)
    s.attrs["cnn_tau"] = tau
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    C.save_json({"cnn_confidence_threshold": tau, "n_images": int(df.image.nunique())}, os.path.join(out, "summary.json"))
    print(s[["eps_255", "changed", "psnr_db", "wm_fgsm", "wm_pgd20", "cnn_fgsm", "cnn_pgd20", "harmful_fgsm", "harmful_pgd20"]].round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--deployed", default="medical_cnn_robust.pth")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = C.out_dir("eps_sweep")
    if args.summarize_only:
        summarize(out); return
    C.seed_everything()
    cfg, secret = W.WMConfig(), W.secret_from_env()
    items = C.list_images("Test Set", None, exclude_dev=True)
    rng = np.random.default_rng(C.SEED + 1)
    pick = []
    for idx in range(len(C.CLASSES)):
        cls_items = [it for it in items if it[1] == idx]
        k = min(args.per_class, len(cls_items))
        pick += [cls_items[i] for i in sorted(rng.choice(len(cls_items), k, replace=False))]
    model = C.load_cnn(args.deployed)
    print(f"[eps] {len(pick)} images, eps = {EPS}")
    rows = []
    for n, (path, label, cname) in enumerate(pick, 1):
        img_id = os.path.relpath(path, C.DATA_DIR)
        ws = W.embed(W.crop_to_blocks(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)), PID, cfg, secret)
        x = C.to_tensor_batch([ws]); y = torch.tensor([label], device=x.device)
        clean_correct = cnn_signals(model, ws)["cnn_pred"] == label
        for eps in EPS:
            for a in ATTACKS:
                if eps == 0:
                    img = ws
                elif a == "fgsm":
                    img = C.to_uint8_list(C.fgsm(model, x, y, eps / 255))[0]
                else:
                    img = C.to_uint8_list(C.pgd(model, x, y, eps / 255, eps / 4 / 255, 20))[0]
                changed = bool((img != ws).any())
                v = W.verify(img, PID, cfg, secret); s = cnn_signals(model, img)
                rows.append(dict(image=img_id, cls=cname, label=label, eps=eps, attack=a, changed=changed,
                                 n_pixels_changed=int((img != ws).any(axis=2).sum()),
                                 psnr=cv2.PSNR(ws, img) if changed else np.nan, nc=v["nc"], flag_wm=not v["intact"],
                                 clean_correct=bool(clean_correct), wrong_after=bool(s["cnn_pred"] != label),
                                 cnn_conf=s["cnn_conf"]))
        if n % 10 == 0 or n == len(pick):
            pd.DataFrame(rows).to_csv(os.path.join(out, "cases.csv"), index=False)
            print(f"  {n}/{len(pick)}")
    summarize(out)


if __name__ == "__main__":
    main()
