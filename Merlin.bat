@echo off
setlocal
title Merlin v1.1.0 — Balloon Quantity Analyzer
cd /d "%~dp0"

:: Check Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python not found on PATH.
    echo  Please install Python 3.11+ from https://www.python.org/downloads/
    echo  and check "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Create venv if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  First-time setup — creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate venv and install/update
echo.
echo  Checking dependencies...
call .venv\Scripts\activate.bat
pip install -e . --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Failed to install dependencies.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)

:: Launch Merlin
echo.
echo  Starting Merlin...
echo  (Close this window or press Ctrl+C to stop)
echo.
start "" http://localhost:8501
merlin
pause
