@echo off
REM Quick Debug Batch Script for BUL API
echo ================================================
echo   BUL API - Quick Debug Tool
echo ================================================
echo.

REM Check for Python
echo Checking for Python...
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Python found in PATH
    python --version
) else (
    echo   [FAIL] Python not found in PATH
    echo.
    echo   Checking common locations...
    
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        echo   [OK] Found: %LOCALAPPDATA%\Programs\Python\Python311\python.exe
        set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        echo   [OK] Found: %LOCALAPPDATA%\Programs\Python\Python312\python.exe
        set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    ) else if exist "C:\Python311\python.exe" (
        echo   [OK] Found: C:\Python311\python.exe
        set PYTHON_PATH=C:\Python311\python.exe
    ) else (
        echo   [FAIL] Python not found in common locations
        echo.
        echo   Please install Python from: https://www.python.org/downloads/
        echo   Make sure to check "Add Python to PATH" during installation
        pause
        exit /b 1
    )
)

echo.
echo ================================================
echo   Checking Dependencies
echo ================================================
echo.

if defined PYTHON_PATH (
    %PYTHON_PATH% -c "import requests" 2>nul && echo   [OK] requests || echo   [FAIL] requests - NOT INSTALLED
    %PYTHON_PATH% -c "import fastapi" 2>nul && echo   [OK] fastapi || echo   [FAIL] fastapi - NOT INSTALLED
    %PYTHON_PATH% -c "import uvicorn" 2>nul && echo   [OK] uvicorn || echo   [FAIL] uvicorn - NOT INSTALLED
    %PYTHON_PATH% -c "import colorama" 2>nul && echo   [OK] colorama || echo   [WARN] colorama - Optional
) else (
    python -c "import requests" 2>nul && echo   [OK] requests || echo   [FAIL] requests - NOT INSTALLED
    python -c "import fastapi" 2>nul && echo   [OK] fastapi || echo   [FAIL] fastapi - NOT INSTALLED
    python -c "import uvicorn" 2>nul && echo   [OK] uvicorn || echo   [FAIL] uvicorn - NOT INSTALLED
    python -c "import colorama" 2>nul && echo   [OK] colorama || echo   [WARN] colorama - Optional
)

echo.
echo ================================================
echo   Checking Server Status
echo ================================================
echo.

curl -s http://localhost:8000/api/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Server is running on http://localhost:8000
) else (
    echo   [FAIL] Server is NOT running
    echo.
    echo   To start the server:
    if defined PYTHON_PATH (
        echo     %PYTHON_PATH% api_frontend_ready.py
    ) else (
        echo     python api_frontend_ready.py
    )
)

echo.
echo ================================================
echo   Checking Test Files
echo ================================================
echo.

if exist "test_api_responses.py" (echo   [OK] test_api_responses.py) else (echo   [FAIL] test_api_responses.py - NOT FOUND)
if exist "test_api_advanced.py" (echo   [OK] test_api_advanced.py) else (echo   [FAIL] test_api_advanced.py - NOT FOUND)
if exist "test_security.py" (echo   [OK] test_security.py) else (echo   [FAIL] test_security.py - NOT FOUND)
if exist "api_frontend_ready.py" (echo   [OK] api_frontend_ready.py) else (echo   [FAIL] api_frontend_ready.py - NOT FOUND)

echo.
echo ================================================
echo   Summary
echo ================================================
echo.
echo   Debug complete! Check the output above for issues.
echo.
echo   Next steps:
echo   1. Install missing dependencies (if any)
echo   2. Start the server (if not running)
echo   3. Run tests: python run_tests_debug.py
echo.
pause









