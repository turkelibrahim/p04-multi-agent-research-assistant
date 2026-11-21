# P04 · Çok Ajanlı Araştırma Asistanı — Windows PowerShell Başlatıcı
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error ".venv bulunamadı. Önce 'python -m venv --system-site-packages .venv' ve '.venv\Scripts\python.exe -m pip install --upgrade-strategy only-if-needed -r requirements.txt' çalıştırın."
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
& $VenvPython -m streamlit run (Join-Path $PSScriptRoot "app.py")
