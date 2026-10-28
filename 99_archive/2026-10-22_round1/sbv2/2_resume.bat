@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Resuming training from the last checkpoint (no re-preprocessing)...
python -m modal run --detach train_sbv2.py::train_main --resume
pause
