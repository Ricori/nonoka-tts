@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo GPT-SoVITS v2ProPlus: prepare -> train SoVITS + GPT -> render ja/zh samples on Modal (L40S).
echo Keep this window open to receive the samples; training itself keeps running in the cloud if it closes.
python -m modal run --detach train_gptsovits.py::train_main
pause
