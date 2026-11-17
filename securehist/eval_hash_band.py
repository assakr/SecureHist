"""Experiment A: cryptographic hash vs. watermark rule, and the ambiguous-band case set for Experiment C.

Regenerates the six cases of eval_system.py for every test image (8,820 cases from 1,470 images):
clean, jpeg90, fgsm, pgd20, pgd20_adapt, patch64 (attacks white-box on the deployed, defended CNN).

The stored record of each image is its watermarked PNG (OpenCV encoder, default settings). For every case
three integrity checks are applied:
  * SHA-256 of the file bytes   : flags if the received file differs from the stored file by one byte
  * SHA-256 of decoded pixels   : flags if any pixel differs
  * watermark rule (ours)       : blind, key only
"clean" is received as a lossless PNG RE-SAVE by a different encoder (Pillow, optimize=True): identical
pixels, different bytes. This is the false-alarm test. All other cases are received as PNG (OpenCV),
except jpeg90 (JPEG bytes).

Cases with 0.30 <= NC < 0.60 (the routing band) are saved as PNG + score map for eval_vlm_band.py.

Usage:  set SECUREHIST_WM_SECRET=...   then
        python securehist/eval_hash_band.py [--limit 3]
        python securehist/eval_hash_band.py --summarize-only
Output: results_v2/hash_baseline/{cases.csv, summary.csv, summary.json}
        results_v2/vlm_band/{band_cases.csv, band_counts.csv, cases/*.png, cases/*_map.npy}
"""
import argparse
import hashlib
import io
import os
import zlib

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import common as C
import wm_core as W
from eval_system import LOSSY, NEGATIVE, PID, POSITIVE, ROUTE_NC, cnn_signals, make_cases

CONDS = NEGATIVE + LOSSY + POSITIVE      # clean, jpeg90, fgsm, pgd20, pgd20_adapt, patch64


def png_cv2(rgb):
    return cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


def png_pil(rgb):
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG", optimize=True); return buf.getvalue()


def jpg(rgb, q=90):
    return cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])[1].tobytes()


def decode(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def summarize(out_hash, out_band):
    df = pd.read_csv(os.path.join(out_hash, "cases.csv"))
    rows, js = [], {"n_images": int(df.image.nunique()), "n_cases": int(len(df))}
    for name, col in (("SHA-256 of file bytes", "flag_file_hash"), ("SHA-256 of decoded pixels", "flag_pixel_hash"),
                      ("Watermark rule (ours)", "flag_wm")):
        r = {"method": name}
        for cond in CONDS:
            g = df[df.condition == cond]
            p, lo, hi = C.wilson(int(g[col].sum()), len(g))
            r[cond] = 100 * p; r[cond + "_lo"] = 100 * lo; r[cond + "_hi"] = 100 * hi; r[cond + "_n"] = len(g)
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out_hash, "summary.csv"), index=False)
    js["pixels_changed_share"] = {c: float(df[df.condition == c].changed.mean()) for c in CONDS}
    C.save_json(js, os.path.join(out_hash, "summary.json"))
    band = df[(df.nc >= ROUTE_NC[0]) & (df.nc < ROUTE_NC[1])]
    counts = {c: int((band.condition == c).sum()) for c in CONDS}
    counts["total"] = int(len(band))
    pd.DataFrame([counts]).to_csv(os.path.join(out_band, "band_counts.csv"), index=False)
    print(pd.DataFrame(rows)[["method"] + CONDS].round(1).to_string(index=False))
    print("band:", counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="images per class (smoke test)")
    ap.add_argument("--deployed", default="medical_cnn_robust.pth")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out_hash, out_band = C.out_dir("hash_baseline"), C.out_dir("vlm_band")
    if args.summarize_only:
        summarize(out_hash, out_band); return
    os.makedirs(os.path.join(out_band, "cases"), exist_ok=True)
    C.seed_everything()
    cfg, secret = W.WMConfig(), W.secret_from_env()
    if not secret:
        print("[WARN] SECUREHIST_WM_SECRET is not set: legacy key mode.")
    items = C.list_images("Test Set", args.limit, exclude_dev=True)
    model = C.load_cnn(args.deployed)
    print(f"[hash] {len(items)} images, deployed={args.deployed}")
    rows, band_rows = [], []
    for n, (path, label, cname) in enumerate(items, 1):
        img_id = os.path.relpath(path, C.DATA_DIR)
        stem = os.path.splitext(os.path.basename(path))[0]
        rng = np.random.default_rng(zlib.crc32(img_id.encode("utf-8")))
        ws = W.embed(W.crop_to_blocks(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)), PID, cfg, secret)
        stored = png_cv2(ws)
        ref_file, ref_pix = sha(stored), sha(decode(stored).tobytes())
        clean_pred = cnn_signals(model, ws)["cnn_pred"]
        cases, _ = make_cases(model, ws, label, rng, cfg, adaptive=True)
        for cond in CONDS:
            img = cases[cond]
            received = png_pil(ws) if cond == "clean" else (jpg(ws) if cond == "jpeg90" else png_cv2(img))
            got = decode(received)
            v = W.verify(got, PID, cfg, secret); s = cnn_signals(model, got)
            row = dict(image=img_id, cls=cname, label=label, condition=cond, nc=v["nc"], flag_frac=v["flag_frac"],
                       score_max=v["score_max"], flag_wm=not v["intact"],
                       flag_file_hash=sha(received) != ref_file, flag_pixel_hash=sha(got.tobytes()) != ref_pix,
                       changed=bool((got != ws).any()),
                       harmful=bool(cond in POSITIVE and clean_pred == label and s["cnn_pred"] != label), **s)
            rows.append(row)
            if ROUTE_NC[0] <= v["nc"] < ROUTE_NC[1]:
                case_id = f"{stem}__{cond}"
                cv2.imwrite(os.path.join(out_band, "cases", case_id + ".png"), cv2.cvtColor(got, cv2.COLOR_RGB2BGR))
                np.save(os.path.join(out_band, "cases", case_id + "_map.npy"), v["score_map"].astype(np.float32))
                band_rows.append(dict(case=case_id, **row))
        if n % 10 == 0 or n == len(items):
            pd.DataFrame(rows).to_csv(os.path.join(out_hash, "cases.csv"), index=False)
            pd.DataFrame(band_rows).to_csv(os.path.join(out_band, "band_cases.csv"), index=False)
            print(f"  {n}/{len(items)}  band cases so far: {len(band_rows)}")
    summarize(out_hash, out_band)


if __name__ == "__main__":
    main()
