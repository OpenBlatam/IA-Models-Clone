@echo off
echo ========================================
echo BUL - Business Universal Language
echo Sistema de IA Avanzado
echo ========================================
echo.

REM Detectar Python disponible
echo Detectando Python...
python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
    echo Python encontrado: python
    goto :start_system
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
    echo Python encontrado: py
    goto :start_system
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python3
    echo Python encontrado: python3
    goto :start_system
)

echo ERROR: Python no encontrado
echo Por favor instale Python desde https://python.org
echo O habilite Python desde Microsoft Store
pause
exit /b 1

:start_system
echo.
echo Iniciando sistema BUL...
echo.

REM Verificar archivos disponibles
if exist "bul_infinite_ai.py" (
    echo Iniciando BUL Infinite AI...
    %PYTHON_CMD% bul_infinite_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_omniversal_ai.py" (
    echo Iniciando BUL Omniversal AI...
    %PYTHON_CMD% bul_omniversal_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_universal_ai.py" (
    echo Iniciando BUL Universal AI...
    %PYTHON_CMD% bul_universal_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_cosmic_ai.py" (
    echo Iniciando BUL Cosmic AI...
    %PYTHON_CMD% bul_cosmic_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_transcendental_ai.py" (
    echo Iniciando BUL Transcendental AI...
    %PYTHON_CMD% bul_transcendental_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_divine_ai.py" (
    echo Iniciando BUL Divine AI...
    %PYTHON_CMD% bul_divine_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_supreme_ai.py" (
    echo Iniciando BUL Supreme AI...
    %PYTHON_CMD% bul_supreme_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_absolute_ai.py" (
    echo Iniciando BUL Absolute AI...
    %PYTHON_CMD% bul_absolute_ai.py --host 0.0.0.0 --port 8000
) else if exist "bul_enhanced.py" (
    echo Iniciando BUL Enhanced...
    %PYTHON_CMD% bul_enhanced.py --host 0.0.0.0 --port 8000
) else if exist "bul_api.py" (
    echo Iniciando BUL API...
    %PYTHON_CMD% bul_api.py --host 0.0.0.0 --port 8000
) else (
    echo ERROR: No se encontraron archivos del sistema BUL
    echo Archivos disponibles:
    dir *.py
    pause
    exit /b 1
)

echo.
echo Sistema BUL iniciado exitosamente!
echo Acceda a: http://localhost:8000
echo Documentación: http://localhost:8000/docs
echo.
pause
