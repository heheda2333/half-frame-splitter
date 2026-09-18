@echo off
rem Half-Frame Photo Splitter - GUI launcher (keep this file ASCII-only)
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto nopython

python "check_env.py"
if errorlevel 1 goto nodeps

python "gui.py"
if errorlevel 1 (
  echo.
  echo [!] GUI exited with an error. See the message above.
  pause
)
exit /b 0

:nopython
echo [!] Python not found in PATH. Please install Python 3.8+ first.
pause
exit /b 1

:nodeps
echo.
pause
exit /b 1
