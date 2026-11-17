"""Experiment D: embedding-domain ablation of the blind fragile scheme.

Variants (same keys, blocks, QIM, 4 carriers per 16x16 Cb block, 8 passes, same decision rule shape):
  dwt_coef  QIM on 2 fixed Haar detail coefficients in each of cH and cV
  dwt_svd   QIM on sigma_1, sigma_2 of cH and cV (ours; published settings, step 8, threshold 0.19)
  dft_mag   QIM on 4 mid-frequency magnitudes of the orthonormal 16x16 block DFT (phase kept)

Calibration (development images only: first 10 images of each of the 8 class folders = 80 images):
  * step: bisection so that mean PSNR matches ours (target = mean PSNR of dwt_svd on the same 80 images)
  * tamper threshold: smallest value in 0.01..0.40 (step 0.01) with 0/80 untouched images flagged
    (3x3 smoothing, 2x2 opening, regions >= 8 blocks, as for ours)
  * NC threshold: 0.50, lowered to 0.8 x (lowest development NC) if that NC is below 0.60
  Ours keeps its published settings; the calibration result for ours is recorded for comparison.

Test (1,470 images): PSNR, SSIM, NC watermarked / unwatermarked, untouched false alarms, JPEG-90 flagged,
FGSM and PGD-20 (eps 8/255, defended CNN) flagged, 64x64 copy-move AUC and F1 (3 sites), 32x32 copy-move
detected (3 sites), embedding time.

Usage:  python securehist/eval_embed_ablation.py                 # calibrate (if needed) + test, all variants
        python securehist/eval_embed_ablation.py --limit 3       # smoke test
        python securehist/eval_embed_ablation.py --summarize-only
Output: results_v2/embed_ablation/{calibration.json, cases.csv, adaptive.csv, summary.csv}
"""
import argparse
import glob
import json
import os
import time
import zlib
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim

import common as C
import wm_core as W
from eval_system import PID
from eval_watermark import jpeg, localization, site_boxes, t_splice

VARIANTS = ["dwt_coef", "dwt_svd", "dft_mag"]
LABELS = {"dwt_coef": "QIM on DWT detail coefficients", "dwt_svd": "QIM on DWT-SVD sigma1, sigma2 (ours)",
          "dft_mag": "QIM on DFT mid-frequency magnitudes"}
SITES = 3


def dev_images():
    out = []
    for split in ("Test Set", "Train and Validation Set"):
        for folder in sorted(glob.glob(os.path.join(C.DATA_DIR, split, "*"))):
            out += sorted(glob.glob(os.path.join(folder, "*.jpeg")))[: C.DEV_PER_CLASS]
    return out


def load(path):
    return W.crop_to_blocks(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


def calibrate(secret, out):
    path = os.path.join(out, "calibration.json")
    if os.path.exists(path):
        return json.load(open(path))
    imgs = [load(p) for p in dev_images()]
    print(f"[calibrate] {len(imgs)} development images")
    ours = W.WMConfig()
    target = float(np.mean([cv2.PSNR(o, W.embed(o, PID, ours, secret)) for o in imgs]))
    cal = {"target_psnr_db": target, "n_dev_images": len(imgs)}
    for var in VARIANTS:
        if var == "dwt_svd":
            step = ours.step
        else:
            lo, hi = 0.25, 32.0
            for _ in range(10):                       # PSNR falls as the step grows
                mid = (lo * hi) ** 0.5
                cfg = W.WMConfig(domain=var, step=mid)
                ps = np.mean([cv2.PSNR(o, W.embed(o, PID, cfg, secret)) for o in imgs[::4]])
                lo, hi = (mid, hi) if ps > target else (lo, mid)
            step = round((lo * hi) ** 0.5, 3)
        cfg = W.WMConfig(domain=var, step=step)
        wms = [W.embed(o, PID, cfg, secret) for o in imgs]
        psnr = float(np.mean([cv2.PSNR(o, w) for o, w in zip(imgs, wms)]))
        ncs = [W.nc_score(w, PID, cfg, secret) for w in wms]
        maps = [W.block_distances(w, PID, cfg, secret)[0] for w in wms]
        thr = None
        for t in np.round(np.arange(0.01, 0.401, 0.01), 2):
            c2 = replace(cfg, tamper_threshold=float(t))
            if not any(W.tamper_map(w, PID, c2, secret)[1].any() for w in wms):
                thr = float(t); break
        nc_int = 0.50 if min(ncs) >= 0.60 else round(0.8 * min(ncs), 3)
        entry = {"step": step, "dev_psnr_db": psnr, "dev_min_nc": float(min(ncs)), "calibrated_threshold": thr,
                 "nc_intact": nc_int}
        if var == "dwt_svd":
            entry.update(used_threshold=ours.tamper_threshold, used_nc_intact=ours.nc_intact,
                         note="published settings kept; calibrated_threshold shows what the same procedure gives")
        else:
            entry.update(used_threshold=thr if thr is not None else 0.40, used_nc_intact=nc_int)
        cal[var] = entry
        print(f"  {var}: step {step}, dev PSNR {psnr:.2f}, min NC {min(ncs):.3f}, threshold {entry['used_threshold']}")
    C.save_json(cal, path)
    return cal


def cfg_for(var, cal):
    e = cal[var]
    return W.WMConfig(domain=var, step=e["step"], tamper_threshold=e["used_threshold"], nc_intact=e["used_nc_intact"])


def summarize(out):
    df = pd.read_csv(os.path.join(out, "cases.csv"))
    cal = json.load(open(os.path.join(out, "calibration.json")))
    rows = []
    for var in VARIANTS:
        g = df[df.variant == var]
        if g.empty:
            continue
        img = g.drop_duplicates("image")
        r = {"variant": var, "embedding": LABELS[var], "step": cal[var]["step"], "threshold": cal[var]["used_threshold"],
             "nc_intact": cal[var]["used_nc_intact"], "n_images": len(img)}
        for col in ("psnr", "ssim", "nc_wm", "nc_unwm", "embed_s"):
            r[col] = float(img[col].mean())
        for col in ("fp_untouched", "flag_jpeg90", "flag_fgsm", "flag_pgd20"):
            p, lo, hi = C.wilson(int(img[col].sum()), len(img))
            r[col], r[col + "_lo"], r[col + "_hi"] = 100 * p, 100 * lo, 100 * hi
        if "flag_adaptive" in img.columns and not img["flag_adaptive"].isna().all():
            p, lo, hi = C.wilson(int(img["flag_adaptive"].sum()), len(img))
            r["flag_adaptive"], r["flag_adaptive_lo"], r["flag_adaptive_hi"] = 100 * p, 100 * lo, 100 * hi
            r["adaptive_nc"] = float(img["adaptive_nc"].mean())
        s64 = g[g.tamper == "splice64"]; s32 = g[g.tamper == "splice32"]
        r["splice64_auc"] = float(s64.auc.mean()); r["splice64_f1"] = float(s64.f1.mean())
        r["splice64_detected"] = 100 * float(s64.detected.mean())
        p, lo, hi = C.wilson(int(s32.detected.sum()), len(s32))
        r["splice32_detected"], r["splice32_detected_lo"], r["splice32_detected_hi"] = 100 * p, 100 * lo, 100 * hi
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "summary.csv"), index=False)
    if "flag_adaptive" in df.columns and not df["flag_adaptive"].isna().all():
        ad_df = df.drop_duplicates(["variant", "image"])[["variant", "image", "cls", "flag_adaptive", "adaptive_nc"]].copy()
        ad_df.to_csv(os.path.join(out, "adaptive.csv"), index=False)
    disp_cols = ["variant", "step", "psnr", "ssim", "nc_wm", "nc_unwm", "fp_untouched", "flag_jpeg90", "flag_fgsm",
                 "flag_pgd20"]
    if "flag_adaptive" in t.columns:
        disp_cols += ["flag_adaptive", "adaptive_nc"]
    disp_cols += ["splice64_auc", "splice64_f1", "splice32_detected", "embed_s"]
    print(t[[c for c in disp_cols if c in t.columns]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--deployed", default="medical_cnn_robust.pth")
    ap.add_argument("--adaptive", action="store_true", help="run adaptive attack column for Table 5")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = C.out_dir("embed_ablation")
    if args.summarize_only:
        summarize(out); return
    C.seed_everything()
    secret = W.secret_from_env()
    cal = calibrate(secret, out)
    model = C.load_cnn(args.deployed)
    items = C.list_images("Test Set", args.limit, exclude_dev=True)
    csv_path = os.path.join(out, "cases.csv")
    df_existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    # If cases.csv already has the baseline experiments (splice64, etc.), compute adaptive attacks in-place
    if args.adaptive and df_existing is not None and not df_existing.empty and "tamper" in df_existing.columns:
        if "flag_adaptive" not in df_existing.columns:
            df_existing["flag_adaptive"] = pd.Series(index=df_existing.index, dtype=object)
            df_existing["adaptive_nc"] = pd.Series(index=df_existing.index, dtype=float)
        done = set(df_existing[~df_existing["flag_adaptive"].isna()][["variant", "image"]].itertuples(index=False, name=None))
        for var in [v for v in args.variants.split(",") if v]:
            cfg = cfg_for(var, cal)
            print(f"[ablation-adaptive] {var}: {cfg}")
            for n, (path, label, cname) in enumerate(items, 1):
                img_id = os.path.relpath(path, C.DATA_DIR)
                if (var, img_id) in done:
                    continue
                o = load(path)
                ws = W.embed(o, PID, cfg, secret)
                x = C.to_tensor_batch([ws]); y = torch.tensor([label], device=x.device)
                def restore(adv):
                    fixed = W.restore_carriers(C.to_uint8_list(adv)[0], ws, cfg)
                    return C.to_tensor_batch([fixed])
                ad_tensor = C.pgd(model, x, y, 8 / 255, 2 / 255, 20, project_fn=restore)
                ad_img = W.restore_carriers(C.to_uint8_list(ad_tensor)[0], ws, cfg)
                v_ad = W.verify(ad_img, PID, cfg, secret)
                mask = (df_existing.variant == var) & (df_existing.image == img_id)
                df_existing.loc[mask, "flag_adaptive"] = not v_ad["intact"]
                df_existing.loc[mask, "adaptive_nc"] = v_ad["nc"]
                if n % 25 == 0 or n == len(items):
                    df_existing.to_csv(csv_path, index=False)
                    print(f"  {var} {n}/{len(items)}")
        df_existing.to_csv(csv_path, index=False)
        summarize(out)
        return

    rows = df_existing.to_dict("records") if df_existing is not None else []
    done = {(r["variant"], r["image"]) for r in rows}
    for var in [v for v in args.variants.split(",") if v]:
        cfg = cfg_for(var, cal)
        print(f"[ablation] {var}: {cfg}")
        for n, (path, label, cname) in enumerate(items, 1):
            img_id = os.path.relpath(path, C.DATA_DIR)
            if (var, img_id) in done:
                continue
            o = load(path)
            t0 = time.time(); ws = W.embed(o, PID, cfg, secret); t_emb = time.time() - t0
            v = W.verify(ws, PID, cfg, secret)
            x = C.to_tensor_batch([ws]); y = torch.tensor([label], device=x.device)
            fg = C.to_uint8_list(C.fgsm(model, x, y, 8 / 255))[0]
            pg = C.to_uint8_list(C.pgd(model, x, y, 8 / 255, 2 / 255, 20))[0]
            base = dict(variant=var, image=img_id, cls=cname, psnr=cv2.PSNR(o, ws),
                        ssim=ssim(o, ws, channel_axis=2, data_range=255), embed_s=t_emb, nc_wm=v["nc"],
                        nc_unwm=W.nc_score(o, PID, cfg, secret), fp_untouched=not v["intact"],
                        flag_jpeg90=not W.verify(jpeg(ws, 90), PID, cfg, secret)["intact"],
                        flag_fgsm=not W.verify(fg, PID, cfg, secret)["intact"],
                        flag_pgd20=not W.verify(pg, PID, cfg, secret)["intact"])
            if args.adaptive:
                def restore(adv):
                    fixed = W.restore_carriers(C.to_uint8_list(adv)[0], ws, cfg)
                    return C.to_tensor_batch([fixed])
                ad_tensor = C.pgd(model, x, y, 8 / 255, 2 / 255, 20, project_fn=restore)
                ad_img = W.restore_carriers(C.to_uint8_list(ad_tensor)[0], ws, cfg)
                v_ad = W.verify(ad_img, PID, cfg, secret)
                base["flag_adaptive"] = not v_ad["intact"]
                base["adaptive_nc"] = v_ad["nc"]
            rng = np.random.default_rng(zlib.crc32(img_id.encode("utf-8")))
            h, w = ws.shape[:2]
            for tname, size in (("splice64", 64), ("splice32", 32)):
                for box in site_boxes(rng, h, w, size, SITES):
                    gt = np.zeros((h, w), bool); gt[box[0]:box[0] + size, box[1]:box[1] + size] = True
                    loc = localization(t_splice(ws, box, None), gt, cfg, secret)
                    rows.append(dict(base, tamper=tname, auc=loc["auc"], f1=loc["f1"], detected=loc["detected"]))
            if n % 25 == 0 or n == len(items):
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"  {var} {n}/{len(items)}")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summarize(out)


if __name__ == "__main__":
    main()
