param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating local virtual environment at $VenvPath"
    & $Python -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Could not find $VenvPython"
}

if (-not $SkipInstall) {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -e ".[dev]"
}

Write-Host ""
Write-Host "Local ASCR environment is ready."
Write-Host "Activate it with:"
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then validate with:"
Write-Host "  python scripts/smoke_test.py"
