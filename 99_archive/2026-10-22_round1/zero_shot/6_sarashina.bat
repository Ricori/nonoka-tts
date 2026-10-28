@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Rendering test_sentences.txt with Sarashina2.2-TTS zero-shot on Modal (L4)...
python -m modal run sarashina_zero_shot.py
pause
