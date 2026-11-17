"""Experiment C: VLM auditors on the ambiguous band only (0.30 <= NC < 0.60).

Input: the band cases saved by eval_hash_band.py (results_v2/vlm_band/band_cases.csv + cases/).
Every auditor answers every band case; all are compared with the threshold rule on identical cases.

Auditors
  rule          NC >= 0.50 -> AUTHENTIC (the watermark's own image-level threshold), numbers only
  lr            logistic regression on the same numbers (NC, flagged share, max score, CNN confidence,
                entropy, margin), out-of-fold by image (GroupKFold)
  qwen          Qwen2-VL-2B, original prompt (tile + map + numbers)
  qwen_fewshot  same, preceded by two worked examples built from development images (not test images):
                one untouched watermarked image (AUTHENTIC) and one adaptive-PGD image (TAMPERED)
  qwen_text     Qwen2-VL-2B, numbers only (no images)
  quilt_tissue  Quilt-LLaVA-7B, H&E tile + numbers (needs convert_quilt_llava.py)
  qwen7b        Qwen2-VL-7B, original prompt (optional; about 6 GB of GPU memory in 4-bit)

Usage:  python securehist/eval_vlm_band.py                         # rule, lr, qwen, qwen_fewshot, qwen_text, quilt_tissue
        python securehist/eval_vlm_band.py --auditors qwen7b        # add the larger model if it fits
        python securehist/eval_vlm_band.py --summarize-only
Output: results_v2/vlm_band/{calls_<auditor>.jsonl, summary_table.csv, summary.json}
"""
import argparse
import json
import os
import time
import zlib

import cv2
import numpy as np
import pandas as pd

import common as C
import wm_core as W
from eval_system import PID, POSITIVE, cnn_signals, make_cases

VLMS = ["qwen", "qwen_fewshot", "qwen_text", "quilt_tissue", "qwen7b"]
DEFAULT = ["qwen", "qwen_fewshot", "qwen_text", "quilt_tissue"]
INPUTS = {"rule": "numbers", "lr": "numbers", "qwen": "tile + map + numbers", "qwen_fewshot": "tile + map + numbers",
          "qwen_text": "numbers only", "quilt_tissue": "tile + numbers", "qwen7b": "tile + map + numbers"}
LABELS = {"rule": "Threshold rule (NC >= 0.50)", "lr": "Logistic-regression fusion",
          "qwen": "Qwen2-VL-2B, original prompt", "qwen_fewshot": "Qwen2-VL-2B, few-shot",
          "qwen_text": "Qwen2-VL-2B, text only", "quilt_tissue": "Quilt-LLaVA-7B, tissue",
          "qwen7b": "Qwen2-VL-7B, original prompt"}
FEATURES = ["nc", "flag_frac", "score_max", "cnn_conf", "cnn_entropy", "cnn_margin"]
DEFAULT_QUILT = os.path.join(C.PROJECT, "models", "quilt-llava-v1.5-7b-hf")


def fewshot_examples(deployed):
    """Two worked examples from development images (never test images)."""
    cfg, secret = W.WMConfig(), W.secret_from_env()
    dev = C.list_images("Test Set", C.DEV_PER_CLASS)             # the 10 development images per class
    model = C.load_cnn(deployed)
    out = []
    for (path, label, cname), cond in ((dev[0], "clean"), (dev[-1], "pgd20_adapt")):
        rng = np.random.default_rng(zlib.crc32(path.encode("utf-8")))
        ws = W.embed(W.crop_to_blocks(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)), PID, cfg, secret)
        img = ws if cond == "clean" else make_cases(model, ws, label, rng, cfg, adaptive=True)[0][cond]
        v = W.verify(img, PID, cfg, secret); s = cnn_signals(model, img)
        verdict = "AUTHENTIC" if cond == "clean" else "TAMPERED"
        reason = ("The integrity map is dark everywhere and NC is in the untouched range."
                  if cond == "clean" else "NC is well below the untouched range and damage is spread over the whole map.")
        out.append(dict(rgb=img, score_map=v["score_map"], nc=v["nc"], flag=v["flag_frac"], pred=C.CLASSES[s["cnn_pred"]],
                        conf=s["cnn_conf"], answer=json.dumps({"verdict": verdict, "confidence": 0.9, "reason": reason})))
        print(f"  few-shot example: {os.path.basename(path)} {cond} NC={v['nc']:.3f} -> {verdict}")
    return out


def load_auditor(name, args):
    import vlm_auditor as V
    if name == "qwen":
        return V.VLMAuditor(prompt_mode="with_numbers")
    if name == "qwen_text":
        return V.VLMAuditor(prompt_mode="text_only")
    if name == "qwen_fewshot":
        return V.VLMAuditor(prompt_mode="fewshot", examples=fewshot_examples(args.deployed))
    if name == "qwen7b":
        return V.VLMAuditor(model_id="Qwen/Qwen2-VL-7B-Instruct", prompt_mode="with_numbers")
    if name == "quilt_tissue":
        if not os.path.isdir(args.quilt_path):
            raise SystemExit(f"Quilt-LLaVA not found at {args.quilt_path}; run convert_quilt_llava.py or drop quilt_tissue.")
        return V.QuiltAuditor(args.quilt_path, mode="tissue")
    raise ValueError(name)


def run_auditor(name, df, args, out):
    log_path = os.path.join(out, f"calls_{name}.jsonl")
    done = set()
    if os.path.exists(log_path):
        done = {json.loads(l)["case"] for l in open(log_path, encoding="utf-8") if l.strip()}
    todo = df[~df.case.isin(done)]
    if todo.empty:
        print(f"[{name}] all {len(df)} band cases done"); return
    import torch
    aud = load_auditor(name, args)
    print(f"[{name}] {getattr(aud, 'model_id', name)}: {len(todo)} band cases")
    with open(log_path, "a", encoding="utf-8") as log:
        for k, r in enumerate(todo.itertuples(), 1):
            img = cv2.cvtColor(cv2.imread(os.path.join(out, "cases", r.case + ".png")), cv2.COLOR_BGR2RGB)
            smap = np.load(os.path.join(out, "cases", r.case + "_map.npy"))
            t0 = time.time()
            res = aud.audit(img, smap, r.nc, r.flag_frac, C.CLASSES[int(r.cnn_pred)], r.cnn_conf)
            log.write(json.dumps({"case": r.case, "condition": r.condition, "t": time.time() - t0, **res}, default=float) + "\n")
            log.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"  [{name}] {k}/{len(todo)}")
    del aud
    torch.cuda.empty_cache()


def lr_oof(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = df.condition.isin(POSITIVE).values
    prob = np.full(len(df), np.nan)
    n_groups = df.image.nunique()
    if len(np.unique(y)) < 2:                 # only one class in the band: the fusion can only predict it
        return np.full(len(df), float(y.mean()))
    if n_groups < 2:
        return prob
    for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(df, groups=df.image):
        if len(np.unique(y[tr])) < 2:
            prob[te] = float(y[tr].mean()); continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(df.iloc[tr][FEATURES], y[tr]); prob[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
    return prob


def summarize(df, out):
    from sklearn.metrics import roc_auc_score
    df = df[df.condition != "jpeg90"].reset_index(drop=True)      # lossy re-save is reported apart, as in Table 4
    y = df.condition.isin(POSITIVE).values
    verdicts, scores = {"rule": np.where(df.nc >= 0.50, "AUTHENTIC", "TAMPERED")}, {"rule": None}
    p = lr_oof(df)
    verdicts["lr"] = np.where(np.isnan(p), "UNPARSED", np.where(p >= 0.5, "TAMPERED", "AUTHENTIC")); scores["lr"] = p
    for a in VLMS:
        path = os.path.join(out, f"calls_{a}.jsonl")
        if not os.path.exists(path):
            continue
        calls = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]).drop_duplicates("case", keep="last")
        c = calls.set_index("case").reindex(df.case)
        verdicts[a] = c.verdict.fillna("MISSING").values
        conf = c.vlm_conf.astype(float).fillna(0.5).values
        scores[a] = np.where(verdicts[a] == "TAMPERED", 1.0, -1.0) * conf
    rule_ok = (verdicts["rule"] == "TAMPERED") == y
    rows, js = [], {"n_band_cases_excluding_jpeg": int(len(df)), "n_attacked": int(y.sum()), "n_clean": int((~y).sum())}
    for a, v in verdicts.items():
        parsed = np.isin(v, ["TAMPERED", "AUTHENTIC"])
        flag = v == "TAMPERED"
        ok = (flag == y) & parsed
        r = {"auditor": LABELS[a], "key": a, "inputs": INPUTS[a], "n": int(len(v))}
        r["authentic"] = 100 * float((v == "AUTHENTIC").mean())
        for key, mask, num in (("fpr", ~y, flag & ~y), ("tpr", y, flag & y), ("accuracy", np.ones_like(y), ok)):
            k_, n_ = int(num[mask].sum()), int(mask.sum())
            pp, lo, hi = C.wilson(k_, n_) if n_ else (np.nan, np.nan, np.nan)
            r[key], r[key + "_lo"], r[key + "_hi"], r[key + "_n"] = 100 * pp, 100 * lo, 100 * hi, n_
        s = scores[a]
        r["auc"] = (float(roc_auc_score(y, np.nan_to_num(s, nan=0.0))) if s is not None and len(np.unique(y)) == 2 else np.nan)
        if a == "rule":
            r["mcnemar_p_vs_rule"] = np.nan
        else:
            n01, n10, pv = C.mcnemar_exact(rule_ok, ok)
            r["mcnemar_p_vs_rule"] = pv; r["only_auditor_right"] = n01; r["only_rule_right"] = n10
        r["unparsed"] = int((~parsed).sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "summary_table.csv"), index=False)
    js["rows"] = rows
    C.save_json(js, os.path.join(out, "summary.json"))
    print(t[["auditor", "n", "authentic", "fpr", "tpr", "accuracy", "auc", "mcnemar_p_vs_rule"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auditors", default=",".join(DEFAULT))
    ap.add_argument("--quilt-path", default=DEFAULT_QUILT)
    ap.add_argument("--deployed", default="medical_cnn_robust.pth")
    ap.add_argument("--sample", type=int, default=0,
                    help="audit a fixed random sample of N band cases, stratified by case type (0 = all)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = C.out_dir("vlm_band")
    path = os.path.join(out, "band_cases.csv")
    if not os.path.exists(path):
        raise SystemExit("Run securehist/eval_hash_band.py first: it saves the band cases.")
    df = pd.read_csv(path)
    if args.sample and args.sample < len(df):
        frac = args.sample / len(df)
        df = pd.concat([g.sample(max(1, round(len(g) * frac)), random_state=C.SEED)
                        for _, g in df.groupby("condition")]).reset_index(drop=True)
    print(f"[band] {len(df)} band cases from {df.image.nunique()} images")
    if not args.summarize_only:
        for a in [a for a in args.auditors.split(",") if a]:
            if a not in VLMS:
                raise SystemExit(f"Unknown auditor {a}; choose from {VLMS}")
            run_auditor(a, df, args, out)
    summarize(df, out)


if __name__ == "__main__":
    main()
