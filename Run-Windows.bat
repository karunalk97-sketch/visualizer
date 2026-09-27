@echo off
rem Double-click to run from source on Windows. First run sets everything up
rem (Python virtual environment + dependencies); later runs start instantly, and
rem the dependencies are refreshed automatically after an update.
setlocal
cd /d "%~dp0"

rem Prefer a Python that has prebuilt packages for everything we need (3.10 - 3.13).
set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY (
    where python >nul 2>nul && (
        python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul && set "PY=python"
    )
)

if not defined PY (
    echo A compatible Python ^(3.10 - 3.13^) was not found. Installing Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo.
    echo Python installed. Please double-click Run-Windows.bat again.
    pause
    exit /b
)

rem The stamp records which pyproject.toml the environment was installed from, so a
rem half-finished install is retried and new dependencies are picked up after an update.
set "STAMP=.venv\.setup-stamp"
for %%F in (pyproject.toml) do set "WANT=%%~tF %%~zF"
set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"

if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: setting up, this takes a minute...
    %PY% -m venv .venv || goto :fail
)
if not "%HAVE%"=="%WANT%" (
    echo Installing / updating what the visualizer needs...
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -e . || goto :fail
    >"%STAMP%" echo %WANT%
)

start "" ".venv\Scripts\pythonw.exe" -m visualizer.main
exit /b

:fail
echo.
echo Setup failed - see the messages above.
pause
