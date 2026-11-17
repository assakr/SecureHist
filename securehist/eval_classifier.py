"""Classifier robustness on the held-out test split (RQ1).

For each image and each (model, attack) it stores the clean and adversarial
prediction, so every reported number can be recomputed:
  robust accuracy  = adv correct / all images
  conditional ASR  = adv wrong / clean-correct images  (the usual attack success rate)
with 95% Wilson intervals, per class, and an exact McNemar test baseline vs defended.

Attacks operate on the CNN input (180x180, values in [0,1]), as in training.

Usage:  python securehist/eval_classifier.py [--limit 20]
Output: results_v2/classifier/{per_image.csv, summary.csv, summary.json}
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

import common as C

ATTACKS = [  # name, kind, eps, alpha, iters
    ("FGSM_eps8", "fgsm", 8 / 255, None, 1),
    ("PGD20_eps8", "pgd", 8 / 255, 2 / 255, 20),
    ("PGD40_eps8", "pgd", 8 / 255, 2 / 255, 40),
    ("PGD100_eps8", "pgd", 8 / 255, 2 / 255, 100),
    ("PGD20_eps16", "pgd", 16 / 255, 2 / 255, 20),
]
MODELS = {"baseline": "medical_cnn.pth", "defended": "medical_cnn_robust.pth"}


def load_batch(items):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((C.CNN_INPUT, C.CNN_INPUT)), transforms.ToTensor()])
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _, _ in items]).to(C.device())
    y = torch.tensor([lab for _, lab, _ in items], device=C.device())
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="images per class (smoke test)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    C.seed_everything()
    items = C.list_images("Test Set", args.limit)
    out = C.out_dir("classifier")
    print(f"[classifier] {len(items)} test images, device={C.device()}")

    rows = []
    for mname, wfile in MODELS.items():
        model = C.load_cnn(wfile)
        for b0 in range(0, len(items), args.batch):
            batch = items[b0:b0 + args.batch]
            x, y = load_batch(batch)
            cp, cc, _ = C.predict(model, x)
            for aname, kind, eps, alpha, iters in ATTACKS:
                C.seed_everything(C.SEED + b0)
                adv = C.fgsm(model, x, y, eps) if kind == "fgsm" else C.pgd(model, x, y, eps, alpha, iters)
                ap_, ac, _ = C.predict(model, adv)
                for k, (p, lab, cname) in enumerate(batch):
                    rows.append(dict(image=os.path.relpath(p, C.DATA_DIR), cls=cname, label=lab, model=mname,
                                     attack=aname, clean_pred=int(cp[k]), clean_conf=float(cc[k]),
                                     adv_pred=int(ap_[k]), adv_conf=float(ac[k])))
            print(f"  {mname}: {min(b0 + args.batch, len(items))}/{len(items)}")
    df = pd.DataFrame(rows)
    df["clean_ok"] = df.clean_pred == df.label
    df["adv_ok"] = df.adv_pred == df.label
    df.to_csv(os.path.join(out, "per_image.csv"), index=False)

    summ = []
    for (m, a, cls), g in list(df.groupby(["model", "attack", "cls"])) + [((m, a, "ALL"), g) for (m, a), g in df.groupby(["model", "attack"])]:
        n, nc = len(g), int(g.clean_ok.sum())
        cond = g[g.clean_ok]
        p_clean = C.wilson(nc, n)
        p_rob = C.wilson(int(g.adv_ok.sum()), n)
        p_asr = C.wilson(int((~cond.adv_ok).sum()), len(cond))
        summ.append(dict(model=m, attack=a, cls=cls, n=n,
                         clean_acc=100 * p_clean[0], clean_lo=100 * p_clean[1], clean_hi=100 * p_clean[2],
                         robust_acc=100 * p_rob[0], robust_lo=100 * p_rob[1], robust_hi=100 * p_rob[2],
                         asr_conditional=100 * p_asr[0], asr_lo=100 * p_asr[1], asr_hi=100 * p_asr[2]))
    s = pd.DataFrame(summ).sort_values(["attack", "cls", "model"])
    s.to_csv(os.path.join(out, "summary.csv"), index=False)

    tests = {}
    for a in df.attack.unique():
        base = df[(df.model == "baseline") & (df.attack == a)].sort_values("image")
        dfd = df[(df.model == "defended") & (df.attack == a)].sort_values("image")
        d_only, b_only, p = C.mcnemar_exact(base.adv_ok.values, dfd.adv_ok.values)
        tests[a] = dict(defended_only_correct=d_only, baseline_only_correct=b_only, p_value=p)
    C.save_json({"n_images": len(items), "mcnemar_robust_accuracy": tests,
                 "note": "attacks on 180x180 CNN input; defended = medical_cnn_robust.pth"},
                os.path.join(out, "summary.json"))
    print(s[s.cls == "ALL"][["model", "attack", "clean_acc", "robust_acc", "asr_conditional"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
