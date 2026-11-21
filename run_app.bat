@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [HATA] .venv bulunamadi. Once README.md icindeki kurulum adimlarini calistirin.
    pause
    exit /b 1
)
set PYTHONPATH=%~dp0src
".venv\Scripts\python.exe" -m streamlit run "%~dp0app.py"
