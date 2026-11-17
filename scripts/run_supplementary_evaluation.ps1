# Supplementary Evaluation: Adaptive column for Table 5 + Key Variation for Table 6
if (-not $env:SECUREHIST_WM_SECRET) { Write-Error "Set $env:SECUREHIST_WM_SECRET first (see README)"; exit 1 }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$startTime = Get-Date
Write-Host "[$startTime] Starting supplementary evaluation..." -ForegroundColor Cyan

# 1. Adaptive attack on ablation domains (~1-2 hours)
Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "Stage 1: Adaptive PGD on ablation domains (eval_embed_ablation.py --adaptive)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
.venv\Scripts\python.exe securehist\eval_embed_ablation.py --adaptive
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage 1 failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 2. Key variation across 5 keys x 1,470 images (~2-3 hours)
Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "Stage 2: Key variation across 5 keys (eval_key_variation.py)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
.venv\Scripts\python.exe securehist\eval_key_variation.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage 2 failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 3. Generate updated tables (Tables 1-6)
Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "Stage 3: Regenerating all tables (make_tables.py)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
.venv\Scripts\python.exe securehist\make_tables.py

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n[$endTime] Supplementary evaluation completed in $($duration.TotalHours.ToString('F2')) hours." -ForegroundColor Green
