@echo off
rem Install dependencies (keep this file ASCII-only)
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [!] Python not found in PATH. Please install Python 3.8+ first.
  pause
  exit /b 1
)
echo Installing Pillow / numpy ...
python -m pip install -r "requirements.txt"
echo.
pause
