@echo off
echo ========================================
echo Robot Movement AI - Iniciar API
echo ========================================
echo.

cd /d "%~dp0"

echo Verificando Python...
python --version
if errorlevel 1 (
    echo ERROR: Python no encontrado
    pause
    exit /b 1
)

echo.
echo Instalando dependencias esenciales...
python -m pip install fastapi uvicorn python-dotenv pydantic httpx websockets aiofiles

echo.
echo Iniciando servidor API en http://127.0.0.1:8010
echo Presiona Ctrl+C para detener
echo.

cd ..
python -m robot_movement_ai.main --host 127.0.0.1 --port 8010

pause



