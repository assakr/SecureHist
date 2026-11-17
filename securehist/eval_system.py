"""End-to-end evaluation: watermark + CNN + routed VLM auditor (RQ4-RQ7).

One unified case set on the held-out test split (development images excluded).
For every image the stored image is the watermarked one; each image yields these cases:

  clean        watermarked image, untouched (PNG)                         negative
  jpeg90       watermarked image re-saved as JPEG Q90 (benign, lossy)      negative, reported apart
  fgsm         FGSM eps=8/255 on the full-resolution image                 positive
  pgd20        PGD-20 eps=8/255, alpha=2/255                               positive
  pgd20_adapt  PGD-20 + key-less adaptive step: after every step the       positive
               sigma_1,2 of every block are copied back from the watermarked image
  patch64      PGD-20 eps=16/255 inside a random 64x64 box                  positive

The attacked classifier is the deployed one (default: defended CNN), white-box.

Routing (fixed in advance, not tuned on these data): a case goes to the VLM when
  the watermark rule and the CNN-confidence rule disagree   (CNN flag = confidence < 0.65)
  or 0.30 <= NC < 0.60                                      (between "absent" and "intact").
Otherwise the watermark decision stands. In addition, a random 4% of all cases are sent
to the VLM to measure "VLM alone". Every VLM call is logged verbatim.

Usage:  set SECUREHIST_WM_SECRET=<your secret>
        python securehist/eval_system.py [--limit 10] [--no-vlm] [--no-adaptive]
        python securehist/eval_system.py --summarize-only      (recompute summary from cases.csv)
Output: results_v2/system/{cases.csv, vlm_calls.jsonl, summary.json, summary_table.csv}
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

PID = "PID-102938"
ROUTE_CONF = 0.65
ROUTE_NC = (0.30, 0.60)
VLM_SAMPLE_RATE = 0.04
POSITIVE = ["fgsm", "pgd20", "pgd20_adapt", "patch64"]
NEGATIVE = ["clean"]
LOSSY = ["jpeg90"]
FEATURES = ["nc", "flag_frac", "score_max", "cnn_conf", "cnn_entropy", "cnn_margin"]


def jpeg90(img):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)


def make_cases(model, ws, label, rng, cfg, adaptive=True):
    """Returns {condition: uint8 image} plus attack metadata."""
    import torch
    x = C.to_tensor_batch([ws])
    y = torch.tensor([label], device=x.device)
    cases, meta = {"clean": ws, "jpeg90": jpeg90(ws)}, {}
    cases["fgsm"] = C.to_uint8_list(C.fgsm(model, x, y, 8 / 255))[0]
    cases["pgd20"] = C.to_uint8_list(C.pgd(model, x, y, 8 / 255, 2 / 255, 20))[0]
    if adaptive:
        def restore(adv):
            fixed = W.restore_singular_values(C.to_uint8_list(adv)[0], ws, cfg)
            return C.to_tensor_batch([fixed])
        adv = C.pgd(model, x, y, 8 / 255, 2 / 255, 20, project_fn=restore)
        cases["pgd20_adapt"] = W.restore_singular_values(C.to_uint8_list(adv)[0], ws, cfg)
    h, w = ws.shape[:2]
    r, c = int(rng.integers(0, h - 64)), int(rng.integers(0, w - 64))
    mask = torch.zeros_like(x[:, :1]); mask[:, :, r:r + 64, c:c + 64] = 1
    cases["patch64"] = C.to_uint8_list(C.pgd(model, x, y, 16 / 255, 2 / 255, 20, mask=mask))[0]
    meta["patch_box"] = (r, c)
    return cases, meta


def cnn_signals(model, img):
    _, conf, p = C.predict(model, C.to_tensor_batch([img]))
    p = p[0].cpu().numpy()
    top2 = np.sort(p)[-2:]
    return dict(cnn_pred=int(p.argmax()), cnn_conf=float(conf[0]), cnn_entropy=float(-(p * np.log(p + 1e-12)).sum()),
                cnn_margin=float(top2[1] - top2[0]))


def route(row):
    wm_flag = not row["wm_intact"]
    cnn_flag = row["cnn_conf"] < ROUTE_CONF
    return bool((wm_flag != cnn_flag) or (ROUTE_NC[0] <= row["nc"] < ROUTE_NC[1]))


# --------------------------------------------------------------------------- summary
def summarize(df: pd.DataFrame, out: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    df["attacked"] = df.condition.isin(POSITIVE)
    core = df[df.condition.isin(POSITIVE + NEGATIVE)].copy()       # lossy JPEG kept apart
    df["flag_wm"] = ~df.wm_intact
    df["flag_always"] = True

    # CNN-confidence and learned fusion: out-of-fold by image (5 folds)
    df["flag_cnn"] = False
    df["flag_lr"] = False
    gkf = GroupKFold(n_splits=5)
    idx_core = core.index.values
    for tr, te in gkf.split(core, groups=core.image):
        tr_i, te_i = idx_core[tr], idx_core[te]
        clean_tr = df.loc[tr_i][df.loc[tr_i].condition == "clean"]
        tau = np.quantile(clean_tr.cnn_conf, 0.05)                 # 5% FPR on clean training folds
        test_imgs = df.loc[te_i].image.unique()
        sel = df.image.isin(test_imgs)
        df.loc[sel, "flag_cnn"] = df.loc[sel, "cnn_conf"] < tau
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        lr.fit(df.loc[tr_i, FEATURES], df.loc[tr_i, "attacked"])
        df.loc[sel, "flag_lr"] = lr.predict(df.loc[sel, FEATURES]).astype(bool)

    has_vlm = df.vlm_verdict.notna().any()
    if has_vlm:
        vlm_flag = df.vlm_verdict.map({"TAMPERED": True, "AUTHENTIC": False})
        df["flag_system"] = np.where(df.routed & vlm_flag.notna(), vlm_flag, df.flag_wm).astype(bool)
        guard = df.routed & (df.vlm_verdict == "AUTHENTIC") & (df.nc < 0.5)
        df["flag_system_guard"] = df.flag_system | guard
    detectors = ["flag_always", "flag_wm", "flag_cnn", "flag_lr"] + (["flag_system", "flag_system_guard"] if has_vlm else [])

    table, summ = [], {"n_images": int(df.image.nunique()), "n_cases": int(len(df)),
                       "routing_rule": f"wm vs CNN(conf<{ROUTE_CONF}) disagree, or {ROUTE_NC[0]}<=NC<{ROUTE_NC[1]}"}
    for det in detectors:
        row = {"detector": det}
        for cond in NEGATIVE + LOSSY + POSITIVE:
            g = df[df.condition == cond]
            k, n = int(g[det].sum()), len(g)
            p, lo, hi = C.wilson(k, n)
            key = ("FPR_" if cond in NEGATIVE + LOSSY else "TPR_") + cond
            row[key] = f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
            summ.setdefault(det, {})[key] = (p, lo, hi, n)
        harmful = df[df.attacked & df.harmful]
        k = int((~harmful[det]).sum())
        summ[det]["unsafe_pass_rate"] = C.wilson(k, len(harmful)) + (len(harmful),)
        row["unsafe_pass"] = f"{k}/{len(harmful)}"
        table.append(row)
    pd.DataFrame(table).to_csv(os.path.join(out, "summary_table.csv"), index=False)

    routed = df[df.routed]
    summ["routing"] = {"routed_cases": int(len(routed)), "routing_rate_all_cases": C.wilson(len(routed), len(df)),
                       "routed_by_condition": routed.condition.value_counts().to_dict(),
                       "routing_rate_clean": C.wilson(int(df[df.condition == "clean"].routed.sum()), int((df.condition == "clean").sum()))}
    if has_vlm:
        # accuracies below exclude the lossy JPEG cases (an edit after watermarking, but not an attack)
        r = routed[routed.vlm_verdict.isin(["TAMPERED", "AUTHENTIC"]) & ~routed.condition.isin(LOSSY)]
        correct = (r.vlm_verdict == "TAMPERED") == r.attacked
        summ["routing"]["vlm_accuracy_on_routed"] = C.wilson(int(correct.sum()), len(r))
        summ["routing"]["wm_rule_accuracy_on_routed"] = C.wilson(int((r.flag_wm == r.attacked).sum()), len(r))
        summ["routing"]["lr_accuracy_on_routed"] = C.wilson(int((r.flag_lr == r.attacked).sum()), len(r))
        summ["routing"]["unparsed"] = int((routed.vlm_verdict == "UNPARSED").sum())
        s = df[df.vlm_sampled & df.vlm_verdict.isin(["TAMPERED", "AUTHENTIC"]) & ~df.condition.isin(LOSSY)]
        if len(s):
            summ["vlm_alone_sample"] = {
                "n": int(len(s)),
                "vlm_accuracy": C.wilson(int(((s.vlm_verdict == "TAMPERED") == s.attacked).sum()), len(s)),
                "wm_rule_accuracy_same_cases": C.wilson(int((s.flag_wm == s.attacked).sum()), len(s)),
                "lr_accuracy_same_cases": C.wilson(int((s.flag_lr == s.attacked).sum()), len(s))}
    clean = df[df.condition == "clean"]
    summ["clean_diagnostic_accuracy"] = C.wilson(int((clean.cnn_pred == clean.label).sum()), len(clean))
    for cond in POSITIVE:
        g = df[df.condition == cond]
        summ.setdefault("attack_effect", {})[cond] = {
            "robust_accuracy": C.wilson(int((g.cnn_pred == g.label).sum()), len(g)),
            "harmful_rate": C.wilson(int(g.harmful.sum()), len(g)),
            "mean_linf_255": float(g.linf.mean())}
    summ["latency_s"] = {k: float(df[k].mean()) for k in ("t_verify", "t_cnn") if k in df}
    if has_vlm and len(routed):
        summ["latency_s"]["t_vlm_per_call"] = float(df.loc[df.vlm_verdict.notna(), "t_vlm"].mean())
    C.save_json(summ, os.path.join(out, "summary.json"))
    df.to_csv(os.path.join(out, "cases_with_decisions.csv"), index=False)
    print(pd.DataFrame(table).to_string(index=False))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="images per class (smoke test)")
    ap.add_argument("--deployed", default="medical_cnn_robust.pth", help="classifier being attacked and used")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--prompt", default="with_numbers", choices=["with_numbers", "images_only"])
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = C.out_dir("system")
    if args.summarize_only:
        summarize(pd.read_csv(os.path.join(out, "cases.csv")), out)
        return

    C.seed_everything()
    cfg = W.WMConfig()
    secret = W.secret_from_env()
    if not secret:
        print("[WARN] SECUREHIST_WM_SECRET is not set: legacy key mode - the adaptive attack is weaker than a real one.")
    items = C.list_images("Test Set", args.limit, exclude_dev=True)
    model = C.load_cnn(args.deployed)
    vlm = None
    if not args.no_vlm:
        from vlm_auditor import VLMAuditor
        vlm = VLMAuditor(prompt_mode=args.prompt)
        print(f"[system] VLM {vlm.model_id} ({vlm.precision}), prompt={args.prompt}")
    print(f"[system] {len(items)} images, deployed={args.deployed}, device={C.device()}")

    rows = []
    log = open(os.path.join(out, "vlm_calls.jsonl"), "w", encoding="utf-8")
    for n, (path, label, cname) in enumerate(items, 1):
        img_id = os.path.relpath(path, C.DATA_DIR)
        rng = np.random.default_rng(zlib.crc32(img_id.encode("utf-8")))
        o = W.crop_to_blocks(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        ws = W.embed(o, PID, cfg, secret)
        clean_pred = cnn_signals(model, ws)["cnn_pred"]
        cases, meta = make_cases(model, ws, label, rng, cfg, adaptive=not args.no_adaptive)
        for cond, img in cases.items():
            t0 = time.time(); v = W.verify(img, PID, cfg, secret); t_ver = time.time() - t0
            t0 = time.time(); s = cnn_signals(model, img); t_cnn = time.time() - t0
            row = dict(image=img_id, cls=cname, label=label, condition=cond, nc=v["nc"], flag_frac=v["flag_frac"],
                       score_max=v["score_max"], wm_intact=v["intact"], t_verify=t_ver, t_cnn=t_cnn,
                       linf=float(np.abs(img.astype(int) - ws.astype(int)).max()),
                       harmful=bool(cond in POSITIVE and clean_pred == label and s["cnn_pred"] != label), **s)
            row["routed"] = route(row)
            row["vlm_sampled"] = bool(rng.random() < VLM_SAMPLE_RATE)
            row["vlm_verdict"] = None
            if vlm is not None and (row["routed"] or row["vlm_sampled"]):
                t0 = time.time()
                res = vlm.audit(img, v["score_map"], v["nc"], v["flag_frac"], C.CLASSES[s["cnn_pred"]], s["cnn_conf"])
                row.update(vlm_verdict=res["verdict"], vlm_conf=res["vlm_conf"], t_vlm=time.time() - t0)
                log.write(json.dumps({"image": img_id, "condition": cond, "routed": row["routed"],
                                      "sampled": row["vlm_sampled"], "attacked": cond in POSITIVE, **res}, default=float) + "\n")
                log.flush()
            rows.append(row)
        if n % 10 == 0 or n == len(items):
            pd.DataFrame(rows).to_csv(os.path.join(out, "cases.csv"), index=False)
            print(f"  {n}/{len(items)}  routed so far: {sum(r['routed'] for r in rows)}")
    log.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "cases.csv"), index=False)
    summarize(df, out)


if __name__ == "__main__":
    main()
