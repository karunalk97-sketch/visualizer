@echo off
rem Double-click to run from source on Windows. First run sets everything up
rem (Python virtual environment + dependencies), later runs start instantly.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"

if not defined PY (
    echo Python was not found. Installing Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo.
    echo Python installed. Please double-click Run-Windows.bat again.
    pause
    exit /b
)

if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: setting up, this takes a minute...
    %PY% -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -e . || goto :fail
)

start "" ".venv\Scripts\pythonw.exe" -m visualizer.main
exit /b

:fail
echo.
echo Setup failed - see the messages above.
pause
