@echo off
echo ========================================
echo Fixing Python Environment
echo ========================================
echo.

REM Remove broken venv from PATH temporarily
set "PATH=%PATH:C:\blatam-academy\venv_ultra_advanced\Scripts;=%"
set "PATH=%PATH:;C:\blatam-academy\venv_ultra_advanced\Scripts=%"

echo Checking for Python...

REM Try python
where python >nul 2>&1
if %errorlevel% equ 0 (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] Found: python
        python --version
        set PYTHON_CMD=python
        goto :found
    )
)

REM Try python3
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    python3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] Found: python3
        python3 --version
        set PYTHON_CMD=python3
        goto :found
    )
)

REM Try py launcher
where py >nul 2>&1
if %errorlevel% equ 0 (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] Found: py launcher
        py --version
        set PYTHON_CMD=py
        goto :found
    )
)

echo.
echo [ERROR] No working Python found!
echo.
echo Please install Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation
echo.
pause
exit /b 1

:found
echo.
echo [1/2] Installing dependencies...
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some dependencies may have failed to install
    echo Check output above for details
)

echo.
echo [2/2] Verifying installation...
%PYTHON_CMD% -c "import requests; print('✓ requests installed')"
%PYTHON_CMD% -c "import json; print('✓ json available')"

echo.
echo ========================================
echo Python environment fixed!
echo ========================================
echo.
echo To run tests:
echo   %PYTHON_CMD% test_api_responses.py
echo   %PYTHON_CMD% test_api_advanced.py
echo   %PYTHON_CMD% test_security.py
echo.
echo Or use: run_all_tests.bat
echo.
pause









