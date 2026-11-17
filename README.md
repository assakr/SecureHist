# SecureHist: blind fragile DWT-SVD watermarking for histopathology

Code for the manuscript *"Blind fragile DWT-SVD watermarking for histopathology"*. The watermark is an invisible, key-bound integrity mark embedded in H&E tiles (LC25000 lung/colon set). A suspect image is verified **blind**, with the secret key and patient ID only (no original or reference copy), and the verifier returns a block-level tamper map.

## Method (`securehist/wm_core.py`)

- RGB → YCrCb; the mark goes into the **Cb** channel, in every 16×16 block.
- 1-level Haar DWT per block, then SVD of the **HL** and **LH** sub-bands.
- QIM on σ₁ and σ₂ of each sub-band (4 bits per block, step 8, 8 embedding passes).
- Per-block key bits = HMAC-SHA-256(secret, patient ID, block position).
- Verification: distance of the recomputed singular values to the expected lattice → 3×3 smoothing → threshold 0.19 → 2×2 opening → minimum region of 8 blocks.
- Image score: NC = 1 − 2·BER (≈0.78 watermarked, ≈0 unwatermarked or wrong key).

## Repository layout

| Path | Contents |
|---|---|
| `securehist/wm_core.py` | Watermark embedding, blind verification, tamper map |
| `securehist/common.py` | Data list, statistics (Wilson/bootstrap CIs, McNemar), CNN loading, attacks |
| `securehist/eval_*.py` | One script per experiment (see `securehist/README_RUN.md`) |
| `securehist/vlm_auditor.py`, `convert_quilt_llava.py` | VLM auditors (Qwen2-VL, Quilt-LLaVA) reported as a negative result |
| `securehist/make_tables.py` | Builds every paper table and figure from `results_v2/` |
| `medical_classifier/` | CNN used for the adversarial-attack experiments |
| `train_adversarial_defense.py` | Adversarial training of the defended classifier |
| `scripts/*.ps1` | Full and supplementary evaluation pipelines (Windows) |
| `legacy/` | Earlier DFT / DWT / DWT-SVD scripts, kept for reference; not used for the paper's numbers |

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
```

Data: download LC25000 and place it in `lung_colon_image_set/` at the repository root. Model weights (`medical_cnn.pth`, `medical_cnn_robust.pth`) are not in the repo; train them with `train_adversarial_defense.py` or contact the authors.

Set a secret key before every run (use the same one for all runs):

```powershell
$env:SECUREHIST_WM_SECRET = "choose-a-long-random-secret"
```

Without a secret the key falls back to the patient ID alone, and anyone who knows the ID could forge a valid watermark.

## Reproducing the results

```powershell
.venv\Scripts\python.exe securehist\eval_classifier.py
.venv\Scripts\python.exe securehist\eval_watermark.py
.venv\Scripts\python.exe securehist\eval_system.py
.\scripts\run_full_evaluation.ps1            # hash baseline, eps sweep, VLM band, embedding ablation
.\scripts\run_supplementary_evaluation.ps1   # adaptive attack on ablations, key variation
.venv\Scripts\python.exe securehist\make_tables.py
```

All numbers in the paper are written by these scripts into `results_v2/`. See `securehist/README_RUN.md` for per-script options, runtimes and smoke-test flags.

## Main results (1,470 test images)

| Metric | Value |
|---|---|
| PSNR / SSIM | 56.06 dB / 0.9988 |
| NC: watermarked / unwatermarked / wrong key | 0.781 / −0.004 / 0.001 |
| False alarms on untouched images | 0.9% |
| Detection of 64-px edits: splice / noise / crop / adversarial patch | 98.7 / 99.5 / 89.3 / 93.5% |
| JPEG, resize, blur flagged | 100% (fragile by design) |
