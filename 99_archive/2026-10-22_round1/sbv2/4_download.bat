@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist out mkdir out
echo Downloading samples...
python -m modal volume get nonoka-sbv2 samples out --force
echo Downloading model files...
python -m modal volume get nonoka-sbv2 model_assets/nonoka out --force
echo Done. Files are in the "out" folder.
pause
