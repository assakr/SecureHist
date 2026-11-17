# Full Evaluation Pipeline for SecureHist v2
# Estimated total runtime: ~6-10 hours on RTX A3000 GPU

if (-not $env:SECUREHIST_WM_SECRET) { Write-Error "Set $env:SECUREHIST_WM_SECRET first (see README)"; exit 1 }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$startTime = Get-Date
Write-Host "[$startTime] Starting full evaluation pipeline..." -ForegroundColor Cyan

# Stage A: Hash vs Watermark (~2-3 hours)
Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "Stage A: Hash baseline vs. Watermark (eval_hash_band.py)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
.venv\Scripts\python.exe securehist\eval_hash_band.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage A failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# Stage B: Perturbation budget sweep (~1 hour)
Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "Stage B: Perturbation budget sweep (eval_eps_sweep.py)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
.venv\Scripts\python.exe securehist\eval_eps_sweep.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage B failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# Stage C: VLM auditors on the ambiguous band (~1-2 hours)
Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "Stage C: VLM auditors on ambiguous band (eval_vlm_band.py)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
.venv\Scripts\python.exe securehist\eval_vlm_band.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage C failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# Stage D: Embedding domain ablation (~2-4 hours)
Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "Stage D: Embedding domain ablation (eval_embed_ablation.py)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
.venv\Scripts\python.exe securehist\eval_embed_ablation.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage D failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# Generate Tables and Figures
Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "Generating final tables and figures (make_tables.py)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
.venv\Scripts\python.exe securehist\make_tables.py

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n[$endTime] Full pipeline completed in $($duration.TotalHours.ToString('F2')) hours." -ForegroundColor Green
