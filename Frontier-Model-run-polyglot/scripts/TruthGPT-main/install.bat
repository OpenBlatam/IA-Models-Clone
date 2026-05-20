@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   TRUTHGPT - USER INSTALLATION (Windows)
echo ============================================================
echo.

:: 1. Check for Python
echo [1/5] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10 or newer from https://www.python.org/downloads/
    echo and make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
python -c "import sys; print(f'Detected Python {sys.version.split()[0]}')"

:: 2. Create Virtual Environment
echo.
echo [2/5] Creating virtual environment (.venv)...
if exist .venv (
    echo     Pending .venv found, skipping creation.
) else (
    python -m venv .venv
    echo     Virtual environment created.
)

:: 3. Upgrade pip
echo.
echo [3/5] Upgrading build tools...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel build

:: 4. Install PyTorch (CUDA 11.8 recommended for most)
echo.
echo [4/5] Installing PyTorch (CUDA support)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: 5. Install TruthGPT Core
echo.
echo [5/5] Installing TruthGPT Optimization Core...
pip install -e optimization_core

echo.
echo ============================================================
echo   INSTALLATION SUCCESSFUL!
echo ============================================================
echo.
echo To run the application, use the 'run.bat' script.
echo.
pause
