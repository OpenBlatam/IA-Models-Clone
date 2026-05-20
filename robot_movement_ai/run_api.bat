@echo off
echo ========================================
echo Starting Robot Movement AI API Server
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing/updating dependencies...
pip install -q fastapi uvicorn python-dotenv

echo.
echo Starting server on http://localhost:8010
echo Press Ctrl+C to stop
echo.

python -m robot_movement_ai.main --host 0.0.0.0 --port 8010

pause



