@echo off
REM Debug Script for TruthGPT Tests
REM Finds Python and runs diagnostic checks

echo ================================================
echo   TruthGPT Test Debug Tool
echo ================================================
echo.

REM Check for Python
echo Checking for Python...
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Test if python actually works
    python --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo   [OK] Python found in PATH
        python --version
        set PYTHON_CMD=python
        goto :check_deps
    ) else (
        echo   [WARN] Python in PATH but not working (Microsoft Store redirect?)
    )
)

echo   [WARN] Python not found in PATH
echo   Checking common locations...

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo   [OK] Found: %LOCALAPPDATA%\Programs\Python\Python311\python.exe
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    goto :check_deps
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo   [OK] Found: %LOCALAPPDATA%\Programs\Python\Python312\python.exe
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :check_deps
)

if exist "C:\Python311\python.exe" (
    echo   [OK] Found: C:\Python311\python.exe
    set PYTHON_CMD=C:\Python311\python.exe
    goto :check_deps
)

echo   [FAIL] Python not found
echo.
echo   Please install Python from: https://www.python.org/downloads/
echo   Make sure to check "Add Python to PATH" during installation
pause
exit /b 1

:check_deps
echo.
echo ================================================
echo   Checking Dependencies
echo ================================================
echo.

%PYTHON_CMD% -c "import torch" 2>nul && echo   [OK] torch || echo   [FAIL] torch - NOT INSTALLED
%PYTHON_CMD% -c "import numpy" 2>nul && echo   [OK] numpy || echo   [FAIL] numpy - NOT INSTALLED
%PYTHON_CMD% -c "import psutil" 2>nul && echo   [OK] psutil || echo   [WARN] psutil - Optional

echo.
echo ================================================
echo   Checking Test Files
echo ================================================
echo.

if exist "tests\test_core.py" (echo   [OK] test_core.py) else (echo   [FAIL] test_core.py - NOT FOUND)
if exist "tests\test_optimization.py" (echo   [OK] test_optimization.py) else (echo   [FAIL] test_optimization.py - NOT FOUND)
if exist "tests\test_models.py" (echo   [OK] test_models.py) else (echo   [FAIL] test_models.py - NOT FOUND)
if exist "tests\test_training.py" (echo   [OK] test_training.py) else (echo   [FAIL] test_training.py - NOT FOUND)
if exist "tests\test_inference.py" (echo   [OK] test_inference.py) else (echo   [FAIL] test_inference.py - NOT FOUND)
if exist "tests\test_monitoring.py" (echo   [OK] test_monitoring.py) else (echo   [FAIL] test_monitoring.py - NOT FOUND)
if exist "tests\test_integration.py" (echo   [OK] test_integration.py) else (echo   [FAIL] test_integration.py - NOT FOUND)
if exist "run_unified_tests.py" (echo   [OK] run_unified_tests.py) else (echo   [FAIL] run_unified_tests.py - NOT FOUND)
if exist "debug_tests.py" (echo   [OK] debug_tests.py) else (echo   [FAIL] debug_tests.py - NOT FOUND)

echo.
echo ================================================
echo   Running Detailed Debug
echo ================================================
echo.

if exist "check_environment.py" (
    echo   Running comprehensive environment check...
    %PYTHON_CMD% check_environment.py 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo   [WARN] Could not run detailed check - Python may not be working
    )
) else if exist "debug_tests.py" (
    %PYTHON_CMD% debug_tests.py 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo   [WARN] Could not run detailed check - Python may not be working
    )
) else (
    echo   [WARN] Debug scripts not found, skipping detailed checks
)

echo.
echo ================================================
echo   Summary
echo ================================================
echo.
echo   Debug complete! Check the output above for issues.
echo.
echo   Next steps:
echo   1. Install missing dependencies (if any)
echo   2. Run tests: %PYTHON_CMD% run_unified_tests.py
echo.
pause

