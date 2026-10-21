@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist venv\Scripts\python.exe (
  echo venv not found - run run_transcribe.bat once first.
  pause
  exit /b 1
)
venv\Scripts\python -m pip install -q pykakasi
venv\Scripts\python cer_samples.py %* || (echo ERROR - copy the messages above and send them to Claude. & pause & exit /b 1)
pause
