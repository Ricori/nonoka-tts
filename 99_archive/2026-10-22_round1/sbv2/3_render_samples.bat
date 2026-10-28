@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Rendering test_sentences.txt with every checkpoint...
python -m modal run train_sbv2.py::samples_main
pause
