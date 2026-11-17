"""Paired comparison of VLM auditors on the SAME cases: Qwen2-VL-2B (general) vs Quilt-LLaVA-7B (pathology).

Every auditor sees every case (not only routed ones), so each gets a full confusion table, and all
auditors are compared with each other and with the watermark rule on identical inputs (McNemar).

Steps (resumable; re-running skips finished work):
  1. cases:  a fixed random sample of test images (dev images excluded), N per class; each gives the
             six cases of eval_system.py (clean, jpeg90, fgsm, pgd20, pgd20_adapt, patch64), saved as PNG
  2. audit:  each auditor runs on all cases, one model in memory at a time; every call logged verbatim
  3. summary

Usage (after SECUREHIST_WM_SECRET is set and convert_quilt_llava.py has been run once):
  python securehist/eval_vlm_compare.py --per-class 3                  # smoke test
  python securehist/eval_vlm_compare.py                                # 30 per class -> 540 cases
  python securehist/eval_vlm_compare.py --auditors quilt_map,quilt_tissue   # skip Qwen
  python securehist/eval_vlm_compare.py --summarize-only
Output: results_v2/vlm_compare/{cases/, cases.csv, calls_<auditor>.jsonl, summary.json, summary_table.csv}
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
from eval_system import LOSSY, NEGATIVE, PID, POSITIVE, cnn_signals, make_cases, route

AUDITORS = ["qwen", "quilt_map", "quilt_tissue"]
DEFAULT_QUILT = os.path.join(C.PROJECT, "models", "quilt-llava-v1.5-7b-hf")


def build_cases(args, out):
    csv_path = os.path.join(out, "cases.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if df.image.nunique() >= 3 * args.per_class:
            print(f"[cases] reusing {len(df)} cases"); return df
    C.seed_everything()
    cfg, secret = W.WMConfig(), W.secret_from_env()
    if not secret:
        print("[WARN] SECUREHIST_WM_SECRET is not set: legacy key mode.")
    items = C.list_images("Test Set", None, exclude_dev=True)
    rng = np.random.default_rng(C.SEED)
    pick = []
    for idx in range(len(C.CLASSES)):
        cls_items = [it for it in items if it[1] == idx]
        pick += [cls_items[i] for i in sorted(rng.choice(len(cls_items), args.per_class, replace=False))]
    model = C.load_cnn(args.deployed)
    os.makedirs(os.path.join(out, "cases"), exist_ok=True)
    rows = []
    for n, (path, label, cname) in enumerate(pick, 1):
        img_id = os.path.relpath(path, C.DATA_DIR)
        rng_i = np.random.default_rng(zlib.crc32(img_id.encode("utf-8")))
        ws = W.embed(W.crop_to_blocks(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)), PID, cfg, secret)
        clean_pred = cnn_signals(model, ws)["cnn_pred"]
        cases, _ = make_cases(model, ws, label, rng_i, cfg, adaptive=True)
        for cond, img in cases.items():
            v = W.verify(img, PID, cfg, secret); s = cnn_signals(model, img)
            case_id = f"{os.path.splitext(os.path.basename(path))[0]}__{cond}"
            cv2.imwrite(os.path.join(out, "cases", case_id + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            np.save(os.path.join(out, "cases", case_id + "_map.npy"), v["score_map"].astype(np.float32))
            row = dict(case=case_id, image=img_id, cls=cname, label=label, condition=cond, nc=v["nc"],
                       flag_frac=v["flag_frac"], wm_intact=v["intact"],
                       harmful=bool(cond in POSITIVE and clean_pred == label and s["cnn_pred"] != label), **s)
            row["routed"] = route(row)
            rows.append(row)
        print(f"  cases {n}/{len(pick)}")
    df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)
    return df


def load_auditor(name, args):
    import vlm_auditor as V
    if name == "qwen":
        return V.VLMAuditor(prompt_mode="with_numbers")
    if not os.path.isdir(args.quilt_path):
        raise SystemExit(f"Quilt-LLaVA not found at {args.quilt_path}. Run securehist/convert_quilt_llava.py first.")
    return V.QuiltAuditor(args.quilt_path, mode=name.split("_")[1])


def run_auditor(name, df, args, out):
    log_path = os.path.join(out, f"calls_{name}.jsonl")
    done = set()
    if os.path.exists(log_path):
        done = {json.loads(l)["case"] for l in open(log_path, encoding="utf-8") if l.strip()}
    todo = df[~df.case.isin(done)]
    if todo.empty:
        print(f"[{name}] all {len(df)} cases done"); return
    import torch
    aud = load_auditor(name, args)
    print(f"[{name}] {aud.model_id} ({aud.precision}): {len(todo)} cases to run")
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


def summarize(df, out, auditors):
    df = df.copy()
    df["attacked"] = df.condition.isin(POSITIVE)
    df["flag_wm"] = ~df.wm_intact.astype(bool)
    dets = {"wm_rule": df.flag_wm}
    parsed = {"wm_rule": np.ones(len(df), bool)}          # an unparsed answer counts as wrong
    extra = {}
    for a in auditors:
        p = os.path.join(out, f"calls_{a}.jsonl")
        if not os.path.exists(p):
            continue
        calls = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]).drop_duplicates("case", keep="last")
        m = df[["case"]].merge(calls[["case", "verdict", "reason", "t"]], on="case", how="left")
        df[f"verdict_{a}"] = m.verdict.values
        dets[a] = (m.verdict == "TAMPERED").values
        parsed[a] = m.verdict.isin(["TAMPERED", "AUTHENTIC"]).values
        v = m.verdict
        extra[a] = {"n_calls": int(v.notna().sum()), "unparsed": int((v == "UNPARSED").sum()),
                    "share_tampered": float((v == "TAMPERED").mean()),
                    "distinct_reasons_share": float(m.reason.nunique() / max(1, m.reason.notna().sum())),
                    "most_common_reason_share": float(m.reason.value_counts(normalize=True).iloc[0]) if m.reason.notna().any() else None,
                    "sec_per_call": float(m.t.mean())}
    core = ~df.condition.isin(LOSSY)
    table, summ = [], {"n_images": int(df.image.nunique()), "n_cases": int(len(df)), "auditors": extra}
    for name, flag in dets.items():
        flag = pd.Series(np.asarray(flag, bool), index=df.index)
        row, s = {"detector": name}, {}
        for cond in NEGATIVE + LOSSY + POSITIVE:
            g = df.condition == cond
            p, lo, hi = C.wilson(int(flag[g].sum()), int(g.sum()))
            key = ("FPR_" if cond in NEGATIVE + LOSSY else "TPR_") + cond
            row[key] = f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"; s[key] = (p, lo, hi)
        correct = (flag == df.attacked) & parsed[name]
        for subset, mask in (("all", core), ("routed", core & df.routed)):
            p, lo, hi = C.wilson(int(correct[mask].sum()), int(mask.sum()))
            row[f"acc_{subset}"] = f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}] (n={int(mask.sum())})"; s[f"acc_{subset}"] = (p, lo, hi, int(mask.sum()))
        summ[name] = s; table.append(row)
    # paired McNemar on non-lossy cases: every auditor vs the watermark rule, and auditors against each other
    names = list(dets); summ["mcnemar_accuracy"] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = ((np.asarray(dets[names[i]], bool) == df.attacked.values) & parsed[names[i]])[core.values]
            b = ((np.asarray(dets[names[j]], bool) == df.attacked.values) & parsed[names[j]])[core.values]
            n01, n10, p = C.mcnemar_exact(a, b)
            summ["mcnemar_accuracy"][f"{names[i]} vs {names[j]}"] = {f"{names[j]}_only_right": n01, f"{names[i]}_only_right": n10, "p": p}
    pd.DataFrame(table).to_csv(os.path.join(out, "summary_table.csv"), index=False)
    df.to_csv(os.path.join(out, "cases_with_verdicts.csv"), index=False)
    C.save_json(summ, os.path.join(out, "summary.json"))
    print(pd.DataFrame(table).to_string(index=False))
    print(json.dumps(extra, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--auditors", default=",".join(AUDITORS))
    ap.add_argument("--quilt-path", default=DEFAULT_QUILT)
    ap.add_argument("--deployed", default="medical_cnn_robust.pth")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    auditors = [a for a in args.auditors.split(",") if a]
    bad = set(auditors) - set(AUDITORS)
    if bad:
        raise SystemExit(f"Unknown auditor(s) {bad}; choose from {AUDITORS}")
    out = C.out_dir("vlm_compare")
    if args.summarize_only:
        summarize(pd.read_csv(os.path.join(out, "cases.csv")), out, AUDITORS); return
    df = build_cases(args, out)
    for a in auditors:
        run_auditor(a, df, args, out)
    summarize(df, out, AUDITORS)


if __name__ == "__main__":
    main()
