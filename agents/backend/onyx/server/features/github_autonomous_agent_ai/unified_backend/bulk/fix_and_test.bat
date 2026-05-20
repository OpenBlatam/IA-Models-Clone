@echo off
echo ========================================
echo BUL API - Fix Environment and Run Tests
echo ========================================
echo.

REM Try to find working Python
echo [1/5] Finding Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Or add Python to your PATH
    pause
    exit /b 1
)

python --version
if %errorlevel% neq 0 (
    echo ERROR: Python found but cannot execute
    pause
    exit /b 1
)

echo.
echo [2/5] Checking dependencies...
python -c "import requests" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing missing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo [3/5] Checking server status...
python -c "import requests; r = requests.get('http://localhost:8000/api/health', timeout=2); print('Server OK') if r.status_code == 200 else print('Server not OK')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Server may not be running
    echo Starting server in background...
    start /B python api_frontend_ready.py
    echo Waiting 5 seconds for server to start...
    timeout /t 5 /nobreak >nul
)

echo.
echo [4/5] Running basic tests...
python test_api_responses.py
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some basic tests failed
)

echo.
echo [5/5] Running advanced tests...
python test_api_advanced.py
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some advanced tests failed
)

echo.
echo [6/6] Running security tests...
python test_security.py
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Security issues found
)

echo.
echo ========================================
echo Tests completed
echo Check results in:
echo - test_results.json
echo - test_results.csv
echo - test_dashboard.html
echo ========================================
pause









