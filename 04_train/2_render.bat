@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Re-rendering test_sentences.txt (ja) and zh_sentences.txt (zh) with the trained GPT-SoVITS weights on Modal...
python -m modal run train_gptsovits.py::render_main || (pause & exit /b 1)
python make_blind.py
python make_blind.py out/zh
pause
