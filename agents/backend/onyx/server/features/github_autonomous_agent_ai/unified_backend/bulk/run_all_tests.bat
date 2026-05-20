@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Ejecutando Todas las Pruebas - API BUL
echo ========================================
echo.

REM Buscar Python en varias ubicaciones
set PYTHON_CMD=
where python >nul 2>&1
if %errorlevel% equ 0 (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
    )
)

REM Si no encontro, intentar python3
if not defined PYTHON_CMD (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        python3 --version >nul 2>&1
        if %errorlevel% equ 0 (
            set PYTHON_CMD=python3
        )
    )
)

REM Si aun no encontro, intentar py launcher
if not defined PYTHON_CMD (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py --version >nul 2>&1
        if %errorlevel% equ 0 (
            set PYTHON_CMD=py
        )
    )
)

REM Verificar Python encontrado
if not defined PYTHON_CMD (
    echo ERROR: Python no encontrado
    echo.
    echo Por favor instala Python desde: https://www.python.org/downloads/
    echo O asegurate de que Python este en tu PATH
    pause
    exit /b 1
)

echo [OK] Python encontrado: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

echo [1/7] Verificando servidor...
REM Usar Python para verificar servidor en lugar de curl
%PYTHON_CMD% -c "import requests; r = requests.get('http://localhost:8000/api/health', timeout=2); exit(0 if r.status_code == 200 else 1)" 2>nul
if %errorlevel% neq 0 (
    echo ADVERTENCIA: Servidor no esta corriendo o no responde
    echo Por favor inicia el servidor en otra ventana:
    echo   %PYTHON_CMD% api_frontend_ready.py
    echo.
    echo Continuando con las pruebas de todas formas...
    echo.
)

echo [2/8] Ejecutando pruebas basicas...
%PYTHON_CMD% test_api_responses.py
set BASIC_EXIT=%errorlevel%
if %BASIC_EXIT% neq 0 (
    echo.
    echo ADVERTENCIA: Algunas pruebas basicas fallaron
)

echo.
echo [2.5/8] Ejecutando pruebas completas...
if exist test_complete_api.py (
    %PYTHON_CMD% test_complete_api.py
    set COMPLETE_EXIT=%errorlevel%
    if %COMPLETE_EXIT% neq 0 (
        echo.
        echo ADVERTENCIA: Algunas pruebas completas fallaron
    )
) else (
    echo ADVERTENCIA: test_complete_api.py no encontrado, saltando...
)

echo.
echo [3/8] Ejecutando pruebas avanzadas...
%PYTHON_CMD% test_api_advanced.py
set ADVANCED_EXIT=%errorlevel%
if %ADVANCED_EXIT% neq 0 (
    echo.
    echo ADVERTENCIA: Algunas pruebas avanzadas fallaron
)

echo.
echo [4/8] Ejecutando pruebas de seguridad...
%PYTHON_CMD% test_security.py
set SECURITY_EXIT=%errorlevel%
if %SECURITY_EXIT% neq 0 (
    echo.
    echo ADVERTENCIA: Se encontraron vulnerabilidades de seguridad
)

echo.
echo [5/8] Ejecutando health check avanzado...
if exist health_check_advanced.py (
    %PYTHON_CMD% health_check_advanced.py
    set HEALTH_EXIT=%errorlevel%
    if %HEALTH_EXIT% neq 0 (
        echo.
        echo ADVERTENCIA: Problemas de salud detectados
    )
) else (
    echo ADVERTENCIA: health_check_advanced.py no encontrado, saltando...
)

echo.
echo [6/8] Generando documentacion...
if exist api_doc_generator.py (
    %PYTHON_CMD% api_doc_generator.py
    set DOC_EXIT=%errorlevel%
    if %DOC_EXIT% neq 0 (
        echo.
        echo ADVERTENCIA: Error generando documentacion
    )
) else (
    echo ADVERTENCIA: api_doc_generator.py no encontrado, saltando...
)

echo.
echo [7/8] Generando dashboard...
if exist test_dashboard_generator.py (
    %PYTHON_CMD% test_dashboard_generator.py
    set DASH_EXIT=%errorlevel%
    if %DASH_EXIT% neq 0 (
        echo.
        echo ADVERTENCIA: Error generando dashboard
    )
) else (
    echo ADVERTENCIA: test_dashboard_generator.py no encontrado, saltando...
)

echo.
echo ========================================
echo Pruebas completadas
echo Revisa los archivos de resultados:
echo - test_results.json
echo - test_results.csv
echo - test_dashboard.html
echo - health_check_results.json
echo ========================================
echo.
echo ========================================
echo Resumen de resultados:
echo ========================================
if %BASIC_EXIT% equ 0 (echo Pruebas basicas: OK) else (echo Pruebas basicas: FALLIDAS)
if defined COMPLETE_EXIT (
    if !COMPLETE_EXIT! equ 0 (echo Pruebas completas: OK) else (echo Pruebas completas: FALLIDAS)
)
if %ADVANCED_EXIT% equ 0 (echo Pruebas avanzadas: OK) else (echo Pruebas avanzadas: FALLIDAS)
if %SECURITY_EXIT% equ 0 (echo Pruebas de seguridad: OK) else (echo Pruebas de seguridad: FALLIDAS)
echo.

echo Abriendo dashboard en el navegador...
if exist test_dashboard.html (
    start test_dashboard.html
) else (
    echo ADVERTENCIA: test_dashboard.html no encontrado
)

echo.
echo Presiona cualquier tecla para salir...
pause >nul

