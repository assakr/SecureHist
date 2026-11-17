# SecureHist v2: reproducible evaluation

Every number in the revised paper must come from a file these scripts write into
`results_v2/`. Nothing is typed by hand, and nothing comes from the older scripts
(`generate_comparison_charts.py`, `generate_sop2_metrics.py`, `result 2/*.md` and the
`model_*.py` pipelines).

## What each script answers

| Script | Paper section | Output |
|---|---|---|
| `eval_classifier.py` | RQ1: attacks on the classifier | `results_v2/classifier/` |
| `eval_watermark.py` | RQ2–RQ3: watermark quality, authentication, false alarms, localization | `results_v2/watermark/` |
| `eval_system.py` | RQ4–RQ7: full system, routing, VLM, baselines, adaptive attack | `results_v2/system/` |

Shared code:
- `wm_core.py` is the blind fragile DWT-SVD watermark. It is vectorised, about 8× faster than `model_dwt_svd_blind.py`, and gives the same results.
- `common.py` holds the data list, statistics (Wilson and bootstrap CIs, McNemar), CNN loading and attacks.
- `vlm_auditor.py` is the Qwen2-VL auditor with a neutral prompt.

## Setup (once)

```powershell
cd "D:\PROGRAMMING\dr eman\project 1"
uv pip install --python .venv\Scripts\python.exe PyWavelets scikit-learn scikit-image transformers accelerate bitsandbytes
$env:SECUREHIST_WM_SECRET = "choose-a-long-random-secret"     # keep it private and the same for all runs
```

- The secret matters. Without it, the watermark key is derived from the patient ID alone. Anyone who knows the ID could then forge a valid watermark, and the adaptive-attack result would be meaningless.
- `bitsandbytes` is optional. Without it, the VLM loads in fp16, which needs about 5 GB of GPU memory.

## Run order

1. **Smoke test first (a few minutes).** It checks that everything loads:
   ```powershell
   .venv\Scripts\python.exe securehist\eval_classifier.py --limit 5
   .venv\Scripts\python.exe securehist\eval_watermark.py --limit 3
   .venv\Scripts\python.exe securehist\eval_system.py --limit 3
   ```
2. **Full runs**, all on the held-out test split:
   ```powershell
   .venv\Scripts\python.exe securehist\eval_classifier.py
   .venv\Scripts\python.exe securehist\eval_watermark.py
   .venv\Scripts\python.exe securehist\eval_system.py
   ```
   The classifier run uses all 1,500 test images. The watermark and system runs use 1,470, because the first 10 images of each class were used to fix the watermark thresholds. Progress is saved every 10–25 images.
3. If a run stops during the summary step, recompute it without re-running the experiments:
   `.venv\Scripts\python.exe securehist\eval_system.py --summarize-only`

## What changed compared with the paper's pipeline

- **Watermark verification is blind.** It uses the key only, never a reference image, so it is a real watermark measure. Every result includes a never-watermarked control.
- **Key derivation.** The key is HMAC-SHA-256(secret, patient ID, block) instead of SHA-256(patient ID, block).
- **Image-level decision.** The decision rule was fixed on development images before the test run:
  - threshold 0.19 on the 3×3-smoothed block distance;
  - 2×2 opening;
  - smallest region 8 blocks;
  - NC ≥ 0.50.
  On the development images this gave 0/80 false alarms on untouched images and flagged 96% of 64×64 copy-move edits.
- **Routing to the VLM is now code.** The rule was fixed in advance and is logged per case.
- **VLM prompt.** It no longer dictates the answer, and the VLM sees no reference-dependent "drift map", because a reference is not available in deployment.
- **Baselines.** Always-flag, watermark rule, CNN confidence, and a logistic-regression fusion of the same signals (out-of-fold by image). Without these, a reviewer cannot judge whether the VLM is needed.
- **Adaptive, key-less attack.** PGD that copies the watermarked image's singular values back after every step.
- **Benign JPEG Q90** is reported separately. The watermark is fragile, so JPEG counts as tampering.
- **Statistics.** Every rate comes with a 95% CI. Every count can be recomputed from the per-image or per-case CSVs.

## Expected run time (NVIDIA GPU)

| Run | Time |
|---|---|
| Classifier | well under an hour |
| Watermark | about 2–3 h (`--no-adv` is faster) |
| System | about 3–5 h with the VLM; the adaptive attack adds about 3 s per image (`--no-adaptive` skips it) |

## Second auditor: Quilt-LLaVA (pathology VLM) vs Qwen2-VL

Quilt-LLaVA v1.5 7B is LLaVA-1.5 instruction-tuned on histopathology (research licence). Its original
code base does not support Windows, so `convert_quilt_llava.py` turns the weights into the standard
`transformers` format once; after that it runs like Qwen.

```powershell
uv pip install --python .venv\Scripts\python.exe sentencepiece protobuf huggingface_hub safetensors
.venv\Scripts\python.exe securehist\convert_quilt_llava.py          # once: ~14 GB download, ~16 GB RAM, writes models\quilt-llava-v1.5-7b-hf
.venv\Scripts\python.exe securehist\eval_vlm_compare.py --per-class 3   # smoke test (54 cases)
.venv\Scripts\python.exe securehist\eval_vlm_compare.py                 # 30 images per class = 540 cases
```

- The same cases go to every auditor: Qwen2-VL-2B (same prompt as before), Quilt with H&E + integrity map in
  one picture (`quilt_map`), and Quilt with the H&E image only plus the numbers (`quilt_tissue`).
- Every auditor answers every case, so each gets false-alarm and detection rates per attack, plus accuracy on
  the cases the router would send it. McNemar tests compare each auditor with the watermark rule and with the others.
- An unreadable answer counts as wrong. The share of TAMPERED answers and the share of repeated reasons are reported,
  to show whether a model is really judging or always giving the same answer.
- Resumable: stop at any time and run the same command again. `--auditors quilt_map,quilt_tissue` skips Qwen.
- Time: about 3 x 540 calls; at 3-6 s per call this is roughly 1.5-3 h on a GPU with 8 GB or more.
- Output: `results_v2/vlm_compare/` (`summary_table.csv`, `summary.json`, `calls_<auditor>.jsonl`).

## New experiments (tables in new_experiments_tables.pdf)

Run in this order; each is resumable or re-runnable, and `make_tables.py` can be run after any of them.

```powershell
$env:SECUREHIST_WM_SECRET = "<the same secret as before>"
.venv\Scripts\python.exe securehist\eval_hash_band.py        # A: hash vs watermark, 8,820 cases; also saves the band cases for C  (~2-3 h)
.venv\Scripts\python.exe securehist\eval_eps_sweep.py        # B: budget sweep, 300 images x 7 eps x 2 attacks                     (~1 h)
.venv\Scripts\python.exe securehist\eval_vlm_band.py         # C: auditors on the 0.30 <= NC < 0.60 band (all band cases)
.venv\Scripts\python.exe securehist\eval_vlm_band.py --auditors qwen7b   # C, optional larger model (about 6 GB GPU in 4-bit)
.venv\Scripts\python.exe securehist\eval_embed_ablation.py   # D: embedding-domain ablation, 3 variants x 1,470 images            (~2-4 h)
.venv\Scripts\python.exe securehist\make_tables.py           # fills all tables + figure -> results_v2\tables\
```

- Smoke tests: `eval_hash_band.py --limit 2`, `eval_eps_sweep.py --per-class 2`, `eval_embed_ablation.py --limit 1`.
  Delete `results_v2\hash_baseline`, `results_v2\vlm_band`, `results_v2\eps_sweep` and `results_v2\embed_ablation` after a smoke test,
  before the full runs (the ablation reuses its calibration file and finished images).
- C audits every band case with 4 auditors. If the band is large (it is mostly adaptive-PGD cases), `--sample 400`
  audits a fixed, stratified sample instead; Table 3 still counts the whole band.
- The few-shot examples are built from development images, never test images.
- D calibrates the step and thresholds of each variant on the 80 development images first (`calibration.json`).
