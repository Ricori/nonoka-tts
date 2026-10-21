@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Rendering zh_sentences.txt with Qwen3-TTS zero-shot (Japanese reference clips) on Modal (6 x L4)...
python -m modal run qwen3_zh.py || (pause & exit /b 1)
python make_blind.py out/zh
start "" "out\zh\blind.html"
