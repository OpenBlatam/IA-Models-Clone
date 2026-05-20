@echo off
echo ========================================
echo BUL API - Run Tests and Debug
echo ========================================
echo.

REM Try to find Python
set PYTHON_CMD=
where python >nul 2>&1
if %errorlevel% equ 0 (
    python --version >nul 2>&1
    if %errorlevel% equ 0 set PYTHON_CMD=python
)

if not defined PYTHON_CMD (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        python3 --version >nul 2>&1
        if %errorlevel% equ 0 set PYTHON_CMD=python3
    )
)

if not defined PYTHON_CMD (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py --version >nul 2>&1
        if %errorlevel% equ 0 set PYTHON_CMD=py
    )
)

if not defined PYTHON_CMD (
    echo ERROR: Python not found
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

echo [1/3] Running quick environment test...
%PYTHON_CMD% quick_test.py
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Environment check failed
    echo.
)

echo.
echo [2/3] Verifying test files...
%PYTHON_CMD% verify_tests.py
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some test files have issues
    echo.
)

echo.
echo [3/3] Running tests and debug...
echo.
echo Running: test_api_responses.py
%PYTHON_CMD% test_api_responses.py
set TEST1=%errorlevel%

echo.
echo Running: test_api_advanced.py
%PYTHON_CMD% test_api_advanced.py
set TEST2=%errorlevel%

echo.
echo Running: test_security.py
%PYTHON_CMD% test_security.py
set TEST3=%errorlevel%

echo.
echo ========================================
echo Test Results Summary
echo ========================================
if %TEST1% equ 0 (echo [OK] test_api_responses.py) else (echo [FAIL] test_api_responses.py)
if %TEST2% equ 0 (echo [OK] test_api_advanced.py) else (echo [FAIL] test_api_advanced.py)
if %TEST3% equ 0 (echo [OK] test_security.py) else (echo [FAIL] test_security.py)
echo.

if %TEST1% equ 0 if %TEST2% equ 0 if %TEST3% equ 0 (
    echo All tests passed!
    exit /b 0
) else (
    echo Some tests failed. Check output above.
    exit /b 1
)









