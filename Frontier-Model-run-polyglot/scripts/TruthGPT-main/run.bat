@echo off
echo ============================================================
echo   TRUTHGPT - RUNNER
echo   Enterprise Edition
echo ============================================================
echo.

if not exist .venv (
    echo [ERROR] Virtual environment not found!
    echo Please run 'install.bat' first.
    pause
    exit /b 1
)

:: Activate environment
call .venv\Scripts\activate.bat

:: Execute
echo Launching TruthGPT Interactive Command Center...
python optimization_core/main.py %*

echo.
pause
