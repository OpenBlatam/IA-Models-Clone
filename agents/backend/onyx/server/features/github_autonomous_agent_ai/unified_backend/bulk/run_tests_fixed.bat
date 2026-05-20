@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Running Tests - BUL API
echo ========================================
echo.

REM Remove broken venv from PATH
set "PATH=%PATH:C:\blatam-academy\venv_ultra_advanced\Scripts;=%"
set "PATH=%PATH:;C:\blatam-academy\venv_ultra_advanced\Scripts=%"

REM Find Python
set PYTHON_CMD=

REM Try python
where python >nul 2>&1
if %errorlevel% equ 0 (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
        goto :found
    )
)

REM Try python3
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    python3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python3
        goto :found
    )
)

REM Try py launcher
where py >nul 2>&1
if %errorlevel% equ 0 (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
        goto :found
    )
)

REM Check common file locations
echo Checking for Python in common locations...
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
) do (
    if exist %%P (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 (
            set PYTHON_CMD=%%P
            goto :found
        )
    )
)

echo.
echo ERROR: Python not found!
echo.
echo Please install Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation
echo.
pause
exit /b 1

:found
echo [OK] Python found: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM Check server
echo [1/5] Checking server...
%PYTHON_CMD% -c "import requests; r = requests.get('http://localhost:8000/api/health', timeout=2); exit(0 if r.status_code == 200 else 1)" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Server not running
    echo Start server with: %PYTHON_CMD% api_frontend_ready.py
    echo.
)

REM Run tests
echo [2/5] Running test_api_responses.py...
%PYTHON_CMD% test_api_responses.py
set TEST1=!errorlevel!

echo.
echo [3/5] Running test_api_advanced.py...
%PYTHON_CMD% test_api_advanced.py
set TEST2=!errorlevel!

echo.
echo [4/5] Running test_security.py...
%PYTHON_CMD% test_security.py
set TEST3=!errorlevel!

echo.
echo [5/5] Summary
echo ========================================
if !TEST1! equ 0 (echo [OK] test_api_responses.py) else (echo [FAIL] test_api_responses.py)
if !TEST2! equ 0 (echo [OK] test_api_advanced.py) else (echo [FAIL] test_api_advanced.py)
if !TEST3! equ 0 (echo [OK] test_security.py) else (echo [FAIL] test_security.py)
echo.

if !TEST1! equ 0 if !TEST2! equ 0 if !TEST3! equ 0 (
    echo All tests passed!
    exit /b 0
) else (
    echo Some tests failed. Check output above.
    exit /b 1
)









