# Dev launcher (Windows): start the vLLM mock + orchestrator API together.
# Usage:  .\nse\scripts\run_dev.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $root "venv\Scripts\python.exe"
$env:PYTHONPATH = $root

Write-Host "Starting vLLM mock on :8080 ..."
Start-Process -FilePath $py -ArgumentList "-m","nse.scripts.vllm_mock" -WindowStyle Hidden

Start-Sleep -Seconds 2
Write-Host "Starting orchestrator API on :8000 ..."
& $py -m uvicorn nse.orchestrator.app:app --host 127.0.0.1 --port 8000
