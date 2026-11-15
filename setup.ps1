# kinesplat one-time setup (Windows, PowerShell).
# Creates a local virtual environment, installs PyTorch and the runtime
# dependencies, then fetches the pretrained depth model.
#
#   .\setup.ps1          auto: CUDA build when an NVIDIA GPU is present, else CPU
#   .\setup.ps1 -Cpu     force the CPU-only build
#   .\setup.ps1 -Gpu     require the CUDA build (fails instead of falling back)

param(
    [switch]$Cpu,
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Cpu -and $Gpu) {
    Write-Error "Choose either -Cpu or -Gpu, not both."
}

$useGpu = $false
if ($Gpu) {
    $useGpu = $true
} elseif (-not $Cpu) {
    $useGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
}

Write-Host ""
Write-Host "  kinesplat setup" -ForegroundColor Green
Write-Host "  --------------" -ForegroundColor DarkGray

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python was not found. Install Python 3.10+ from https://python.org and tick 'Add to PATH'."
}

$venv = Join-Path $PSScriptRoot ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "  Creating virtual environment in .venv ..." -ForegroundColor Cyan
    & python -m venv $venv
}

Write-Host "  Updating pip ..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip --quiet

if ($useGpu) {
    Write-Host "  Installing CUDA PyTorch 2.11 (one-time ~2.5 GB download) ..." -ForegroundColor Cyan
    & $py -m pip install --upgrade --force-reinstall "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed. Fix the CUDA/Python environment or rerun explicitly with -Cpu."
    }
} else {
    Write-Host "  Installing CPU PyTorch ..." -ForegroundColor Cyan
    & $py -m pip install --upgrade "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "  Installing dependencies ..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

if ($useGpu) {
    $cudaAvailable = & $py -c "import torch; print('true' if torch.cuda.is_available() else 'false')"
    if ($LASTEXITCODE -ne 0 -or ($cudaAvailable | Select-String -Quiet -Pattern '^true$') -eq $false) {
        throw "CUDA PyTorch installed but torch.cuda.is_available() is false; refusing to run TripoSR on CPU."
    }
}

Write-Host "  Checking the pretrained depth model ..." -ForegroundColor Cyan
& $py -m splatlab.cli model
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Could not download the depth model now; it will be fetched on first use." -ForegroundColor Yellow
}

Write-Host ""
& $py -c "import torch; print('  PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host "  Start the studio with:  .\run.ps1" -ForegroundColor White
Write-Host ""
