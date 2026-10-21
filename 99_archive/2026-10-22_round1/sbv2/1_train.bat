@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Uploading data and starting training on Modal (L40S). It keeps running in the cloud even if you close this window.
python -m modal run --detach train_sbv2.py::train_main
pause
