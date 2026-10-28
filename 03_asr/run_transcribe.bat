@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo [1/3] Creating virtual environment...
  python -m venv venv || goto :err
)
echo [2/3] Installing packages (first time ~1-2 GB, a few minutes)...
venv\Scripts\python -m pip install -q --upgrade pip
venv\Scripts\python -m pip install -q faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10" soundfile scipy || goto :err
echo [3/3] Transcribing (first time downloads ~4.5 GB of models)...
set TARGET=%~1
if "%TARGET%"=="" set TARGET=..\02_dataset_v1
venv\Scripts\python transcribe.py "%TARGET%" || goto :err
echo.
echo Finished. Tell Claude it is done.
pause
exit /b 0
:err
echo.
echo ERROR - copy the messages above and send them to Claude.
pause
exit /b 1
