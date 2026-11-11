@echo off
setlocal
cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo PROXIMAL FLAP could not start. Make sure Python and requirements are installed.
    echo Install dependencies with: python -m pip install -r requirements.txt
    pause
)
