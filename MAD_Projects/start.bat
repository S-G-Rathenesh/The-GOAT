@echo off
title THE GOAT - AI Weight Estimation and Anomaly Detection System

REM Navigate to script directory
cd /d "%~dp0"

REM If in workspace root, navigate into MAD_Projects
if exist "MAD_Projects\app.py" cd /d "%~dp0MAD_Projects"

echo ===============================================================================
echo                 THE GOAT - AI LIVESTOCK TRADING PLATFORM
echo       AI Fair-Weight Estimation and Temporary Anomaly Detection System
echo ===============================================================================
echo.
echo [*] Working Directory: %CD%

REM Check if Python virtual environment exists
if exist ".venv\Scripts\python.exe" goto :run_app

echo.
echo [*] Virtual environment not found. Setting up Python virtual environment...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found on your system PATH.
    echo Please install Python 3.12 and ensure 'Add Python to PATH' is selected.
    echo.
    pause
    exit /b 1
)

echo [*] Creating virtual environment (.venv)...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [*] Installing dependencies...
if exist "F:\MAD Requirements" (
    echo [*] Installing offline from F:\MAD Requirements...
    .venv\Scripts\pip.exe install --no-index --find-links="F:\MAD Requirements" -r requirements.txt
) else (
    echo [*] Installing online from requirements.txt...
    .venv\Scripts\pip.exe install -r requirements.txt
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Setup completed successfully!
echo.

:run_app
if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Streamlit executable not found in .venv\Scripts\streamlit.exe.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo   STARTING FULL-STACK APPLICATION
echo   - Frontend: Streamlit Interactive UI (Desktop and Mobile-Optimized)
echo   - Backend:  OpenCV Silhouette Segmentation and ML Weight Engine
echo ===============================================================================
echo.
echo [*] Local URL: http://localhost:8501
echo [*] Opening default web browser...
echo [*] Press Ctrl+C in this console window to stop the application.
echo.

REM Launch browser after 2 seconds in background
start "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://localhost:8501"

REM Run Streamlit server
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless false

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Server exited with code %ERRORLEVEL%.
    pause
)
