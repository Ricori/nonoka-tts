@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python make_blind.py || (pause & exit /b 1)
python make_blind.py out/zh || (pause & exit /b 1)
start "" "out\blind.html"
start "" "out\zh\blind.html"
