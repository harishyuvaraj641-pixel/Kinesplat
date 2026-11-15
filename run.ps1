# Start the kinesplat local studio.
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [string]$Device = "auto"   # auto | cpu | cuda | mps
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "kinesplat is not set up yet. Run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

$serverArgs = @("-m", "splatlab.server", "--port", $Port, "--device", $Device)
if (-not $NoBrowser) { $serverArgs += "--open" }
& $py @serverArgs
