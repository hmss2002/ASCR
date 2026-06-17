$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$Activate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "Missing .venv. Create it first with:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/setup/bootstrap_local.ps1"
    exit 2
}

. $Activate
Write-Host "Activated ASCR local environment at $ProjectRoot\.venv"
Write-Host "Try: python scripts/smoke_test.py"
