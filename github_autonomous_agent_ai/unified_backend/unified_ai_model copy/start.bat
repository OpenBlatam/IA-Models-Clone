@echo off
REM ============================================
REM Unified AI Model - Quick Start (Windows)
REM ============================================

echo.
echo ============================================
echo  UNIFIED AI MODEL - Starting...
echo ============================================
echo.

REM Check if API key is set
if "%OPENROUTER_API_KEY%"=="" (
    echo [WARNING] OPENROUTER_API_KEY not set!
    echo.
    echo Set it with: set OPENROUTER_API_KEY=sk-or-v1-your-key
    echo.
)

REM Set default model to DeepSeek if not set
if "%UNIFIED_AI_DEFAULT_MODEL%"=="" (
    set UNIFIED_AI_DEFAULT_MODEL=deepseek/deepseek-chat
)

REM Set port if not set
if "%UNIFIED_AI_PORT%"=="" (
    set UNIFIED_AI_PORT=8050
)

echo Model: %UNIFIED_AI_DEFAULT_MODEL%
echo Port: %UNIFIED_AI_PORT%
echo.

REM Run the server
python run.py

pause



