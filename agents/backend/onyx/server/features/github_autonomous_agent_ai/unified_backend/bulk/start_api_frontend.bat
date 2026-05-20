@echo off
echo ========================================
echo BUL API - Frontend Ready
echo ========================================
echo.

REM Detectar Python disponible
echo Detectando Python...
python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
    echo Python encontrado: python
    goto :start_api
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
    echo Python encontrado: py
    goto :start_api
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python3
    echo Python encontrado: python3
    goto :start_api
)

echo ERROR: Python no encontrado
echo Por favor instale Python desde https://python.org
pause
exit /b 1

:start_api
echo.
echo Iniciando BUL API para Frontend...
echo.
echo API disponible en: http://localhost:8000
echo Documentacion: http://localhost:8000/api/docs
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

%PYTHON_CMD% api_frontend_ready.py --host 0.0.0.0 --port 8000

pause
































