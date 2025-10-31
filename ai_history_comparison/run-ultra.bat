@echo off
REM ULTRA OPTIMIZED RUN SCRIPT FOR WINDOWS
REM Script de ejecución ultra optimizado para Windows

echo 🚀 Starting ULTRA OPTIMIZED AI History Comparison System...
echo ⚡ Maximum Performance Mode Activated

REM Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Verificar dependencias ultra
echo 🔍 Checking ultra dependencies...
python -c "import fastapi, uvicorn, loguru" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing ultra dependencies...
    pip install fastapi uvicorn[standard] loguru
)

REM Verificar dependencias de performance
python -c "import uvloop, httptools" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚡ Installing performance boosters...
    pip install uvloop httptools
)

REM Verificar archivo principal
if not exist "ULTRA_OPTIMIZED.py" (
    echo ❌ ULTRA_OPTIMIZED.py not found!
    pause
    exit /b 1
)

echo ✅ All dependencies ready
echo 🚀 Starting ultra optimized server...

REM Configurar variables de entorno
set PYTHONPATH=%PYTHONPATH%;%CD%
set UVICORN_WORKERS=8
set UVICORN_LOOP=uvloop
set UVICORN_HTTP=httptools

REM Ejecutar ultra optimizado
uvicorn ULTRA_OPTIMIZED:app --host 0.0.0.0 --port 8000 --workers 8 --loop uvloop --http httptools --log-level warning --no-access-log --no-server-header --no-date-header --lifespan off

echo 🛑 Ultra optimized server stopped
pause







