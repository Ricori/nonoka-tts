@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
  echo Usage: drop a song file onto 4_sing.bat, or
  echo   4_sing.bat "D:\path\song.mp3" [--start 60 --length 30] [--shift 12] [--refs refcat,ref4690]
  echo Output: 04_train\out\sing\^<song^>\^<run^>\refcat_mix.mp3  ^(README 5.9^)
  pause
  exit /b 1
)
set SONG=%~f1
shift
set ARGS=
:collect
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
echo Converting "%SONG%" to her voice on Modal (stems, pitch, Seed-VC, mix)...
python -m modal run sing_vc.py::main --song "%SONG%"%ARGS% || (echo ERROR - copy the messages above and send them to Claude. & pause & exit /b 1)
pause
