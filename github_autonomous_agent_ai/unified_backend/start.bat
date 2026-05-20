@echo off
REM Unified Backend - Start Script
REM Backend built for Unified AI Model

echo ========================================
echo    Unified Backend - AI Model API
echo ========================================

REM Set Python path
set PYTHONPATH=%~dp0

REM Check for API key
if not defined OPENROUTER_API_KEY (
    if not defined DEEPSEEK_API_KEY (
        echo WARNING: No API key configured
        echo Set OPENROUTER_API_KEY or DEEPSEEK_API_KEY for LLM functionality
        echo.
    )
)

REM Start the server
echo Starting server on http://0.0.0.0:8080
echo Docs available at http://localhost:8080/docs
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
