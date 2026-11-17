"""Blind fragile DWT-SVD watermark on the held-out test split (RQ2-RQ3).

Per image:
  * quality: PSNR, SSIM, embed / verify time
  * authentication: blind NC for watermarked, never-watermarked, and wrong-key images
  * benign processing (false alarms): PNG re-save, JPEG 95/90/75, 0.5x resize, +2 brightness, blur
  * localization: local tampers on the watermarked image AND on the original (control),
    3 random sites each: copy-move 32 and 64 px, local noise 64 px, local crop-resize 64 px,
    adversarial patch 64 px (PGD inside the patch against the CNN; needs PyTorch)

Usage:  set SECUREHIST_WM_SECRET=<your secret>   (Windows)   then
        python securehist/eval_watermark.py [--limit 20] [--no-adv]
Output: results_v2/watermark/{quality.csv, auth.csv, benign.csv, tamper.csv, summary.json}
"""
import argparse
import os
import zlib

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score

import common as C
import wm_core as W

PID = "PID-102938"
SITES = 3


def load_rgb(path):
    return W.crop_to_blocks(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


def jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)


def png(img):
    ok, enc = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)


BENIGN = {
    "png_resave": png,
    "jpeg95": lambda x: jpeg(x, 95),
    "jpeg90": lambda x: jpeg(x, 90),
    "jpeg75": lambda x: jpeg(x, 75),
    "resize_0.5x": lambda x: cv2.resize(cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                        (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR),
    "brightness_+2": lambda x: np.clip(x.astype(int) + 2, 0, 255).astype(np.uint8),
    "blur_0.5": lambda x: cv2.GaussianBlur(x, (0, 0), 0.5),
}


def site_boxes(rng, h, w, size, n):
    return [(int(rng.integers(0, h - size)), int(rng.integers(0, w - size)), size) for _ in range(n)]


def t_splice(img, box, rng):
    r, c, s = box
    h, w = img.shape[:2]
    rs, cs = (r + h // 2) % (h - s), (c + w // 2) % (w - s)
    out = img.copy()
    out[r:r + s, c:c + s] = img[rs:rs + s, cs:cs + s]
    return out


def t_noise(img, box, rng):
    r, c, s = box
    out = img.copy()
    out[r:r + s, c:c + s] = np.clip(img[r:r + s, c:c + s] + rng.normal(0, 15, (s, s, 3)), 0, 255).astype(np.uint8)
    return out


def t_crop(img, box, rng):
    r, c, s = box
    out = img.copy()
    roi = img[r:r + s, c:c + s]
    m = max(1, s // 16)
    inner = cv2.resize(roi[m:-m, m:-m], (s, s), interpolation=cv2.INTER_LINEAR)
    out[r:r + s, c:c + s] = cv2.addWeighted(roi, 0.25, inner, 0.75, 0)
    return out


TAMPERS = {"splice32": (t_splice, 32), "splice64": (t_splice, 64), "noise64": (t_noise, 64), "crop64": (t_crop, 64)}


class AdvPatch:
    """PGD (20 steps, eps 16/255) inside a 64x64 box against the baseline CNN."""

    def __init__(self):
        self.model = C.load_cnn("medical_cnn.pth")

    def __call__(self, img, box, label):
        import torch
        r, c, s = box
        x = C.to_tensor_batch([img])
        mask = torch.zeros_like(x[:, :1])
        mask[:, :, r:r + s, c:c + s] = 1
        y = torch.tensor([label], device=x.device)
        adv = C.pgd(self.model, x, y, 16 / 255, 2 / 255, 20, mask=mask)
        out = C.to_uint8_list(adv)[0]
        flipped = int(C.predict(self.model, C.to_tensor_batch([out]))[0][0]) != label
        return out, flipped


def localization(img, gt_px, cfg, secret):
    score, flagged = W.tamper_map(img, PID, cfg, secret)
    gt = W.block_mask_from_pixels(gt_px, cfg.block)
    tp = int((flagged & gt).sum()); fp = int((flagged & ~gt).sum()); fn = int((~flagged & gt).sum())
    auc = roc_auc_score(gt.ravel(), score.ravel()) if gt.any() and not gt.all() else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(auc=auc, precision=prec, recall=rec, f1=f1, detected=bool(flagged.any()),
                fp_block_rate=fp / max(1, int((~gt).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="images per class (smoke test)")
    ap.add_argument("--no-adv", action="store_true", help="skip the CNN adversarial patch (no PyTorch)")
    ap.add_argument("--include-dev", action="store_true", help="also use the 10 development images per class")
    args = ap.parse_args()
    cfg = W.WMConfig()
    secret = W.secret_from_env()
    if not secret:
        print("[WARN] SECUREHIST_WM_SECRET is not set: legacy key mode (patient ID only) - not secure.")
    items = C.list_images("Test Set", args.limit, exclude_dev=not args.include_dev)
    out = C.out_dir("watermark")
    adv = None if args.no_adv else AdvPatch()
    print(f"[watermark] {len(items)} images, secret={'set' if secret else 'NOT set'}, adv_patch={'on' if adv else 'off'}")

    q, a, b, t = [], [], [], []
    timer = C.Timer()
    for n, (path, label, cname) in enumerate(items, 1):
        img_id = os.path.relpath(path, C.DATA_DIR)
        o = load_rgb(path)
        timer.lap(); ws = W.embed(o, PID, cfg, secret); t_emb = timer.lap()
        v = W.verify(ws, PID, cfg, secret); t_ver = timer.lap()
        q.append(dict(image=img_id, cls=cname, psnr=cv2.PSNR(o, ws), ssim=ssim(o, ws, channel_axis=2, data_range=255),
                      embed_s=t_emb, verify_s=t_ver))
        wrong = W.nc_score(ws, PID, cfg, secret + b"-wrong") if secret else W.nc_score(ws, "PID-000000", cfg)
        a.append(dict(image=img_id, cls=cname, nc_watermarked=v["nc"], intact_watermarked=v["intact"],
                      flag_frac_watermarked=v["flag_frac"], nc_unwatermarked=W.nc_score(o, PID, cfg, secret),
                      nc_wrong_key=wrong))
        for name, fn in BENIGN.items():
            vb = W.verify(fn(ws), PID, cfg, secret)
            b.append(dict(image=img_id, cls=cname, processing=name, nc=vb["nc"], flag_frac=vb["flag_frac"],
                          false_alarm=not vb["intact"]))
        rng = np.random.default_rng(zlib.crc32(img_id.encode('utf-8')))
        h, w = ws.shape[:2]
        jobs = [(k, fn, sz) for k, (fn, sz) in TAMPERS.items()] + ([("advpatch64", None, 64)] if adv else [])
        for tname, fn, size in jobs:
            for box in site_boxes(rng, h, w, size, SITES):
                gt = np.zeros((h, w), bool); gt[box[0]:box[0] + size, box[1]:box[1] + size] = True
                for cond, base in (("watermarked", ws), ("unwatermarked_control", o)):
                    flipped = np.nan
                    if fn is None:
                        tam, flipped = adv(base, box, label)
                    else:
                        tam = fn(base, box, np.random.default_rng(box[0] * 7919 + box[1]))
                    t.append(dict(image=img_id, cls=cname, tamper=tname, condition=cond, r=box[0], c=box[1],
                                  cnn_label_flipped=flipped, **localization(tam, gt, cfg, secret)))
        if n % 25 == 0 or n == len(items):
            print(f"  {n}/{len(items)}")
            for df_, name in ((q, "quality"), (a, "auth"), (b, "benign"), (t, "tamper")):
                pd.DataFrame(df_).to_csv(os.path.join(out, f"{name}.csv"), index=False)

    Q, A, B, T = (pd.DataFrame(x) for x in (q, a, b, t))
    summ = {"n_images": len(items), "dev_images_excluded": not args.include_dev, "secret_mode": "hmac" if secret else "legacy", "config": cfg.__dict__}
    summ["psnr_db"] = C.bootstrap_ci(Q.psnr); summ["ssim"] = C.bootstrap_ci(Q.ssim)
    summ["embed_s"] = float(Q.embed_s.mean()); summ["verify_s"] = float(Q.verify_s.mean())
    for col in ("nc_watermarked", "nc_unwatermarked", "nc_wrong_key"):
        summ[col] = C.bootstrap_ci(A[col])
    summ["clean_watermarked_false_alarm"] = C.wilson(int((~A.intact_watermarked).sum()), len(A))
    summ["min_nc_watermarked"] = float(A.nc_watermarked.min())
    summ["max_nc_unwatermarked_or_wrong_key"] = float(max(A.nc_unwatermarked.max(), A.nc_wrong_key.max()))
    summ["benign_false_alarm"] = {p: C.wilson(int(g.false_alarm.sum()), len(g)) for p, g in B.groupby("processing")}
    summ["localization"] = {}
    for (tn, cond), g in T.groupby(["tamper", "condition"]):
        summ["localization"][f"{tn}|{cond}"] = {m: C.bootstrap_ci(g[m].dropna(), g.dropna(subset=[m]).image)
                                                for m in ("auc", "precision", "recall", "f1", "fp_block_rate")}
        summ["localization"][f"{tn}|{cond}"]["image_detected"] = C.wilson(int(g.detected.sum()), len(g))
        if tn == "advpatch64":
            summ["localization"][f"{tn}|{cond}"]["cnn_flip_rate"] = C.wilson(int(g.cnn_label_flipped.sum()), len(g))
    C.save_json(summ, os.path.join(out, "summary.json"))
    print(f"PSNR {summ['psnr_db'][0]:.2f} dB | NC wm {summ['nc_watermarked'][0]:.3f} | unwm {summ['nc_unwatermarked'][0]:.3f}")


if __name__ == "__main__":
    main()
