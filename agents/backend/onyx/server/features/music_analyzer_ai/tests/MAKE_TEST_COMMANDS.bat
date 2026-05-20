@echo off
REM Script de utilidades para ejecutar tests en Windows
REM Uso: MAKE_TEST_COMMANDS.bat [comando]

if "%1"=="all" (
    echo Ejecutando todos los tests...
    pytest -v
    goto :end
)

if "%1"=="coverage" (
    echo Ejecutando tests con cobertura...
    pytest --cov=music_analyzer_ai --cov-report=html --cov-report=term
    echo Reporte HTML generado en htmlcov\index.html
    goto :end
)

if "%1"=="fast" (
    echo Ejecutando tests rapidos (sin slow)...
    pytest -v -m "not slow"
    goto :end
)

if "%1"=="unit" (
    echo Ejecutando tests unitarios...
    pytest -v -m unit
    goto :end
)

if "%1"=="integration" (
    echo Ejecutando tests de integracion...
    pytest -v -m integration
    goto :end
)

if "%1"=="api" (
    echo Ejecutando tests de API...
    pytest -v -m api tests\test_api.py
    goto :end
)

if "%1"=="parallel" (
    echo Ejecutando tests en paralelo...
    pytest -n auto -v
    goto :end
)

if "%1"=="failed" (
    echo Ejecutando solo tests que fallaron...
    pytest --lf -v
    goto :end
)

if "%1"=="clean" (
    echo Limpiando cache y archivos temporales...
    pytest --cache-clear
    if exist __pycache__ rmdir /s /q __pycache__
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    for /r . %%f in (*.pyc) do del /q "%%f"
    if exist .pytest_cache rmdir /s /q .pytest_cache
    if exist htmlcov rmdir /s /q htmlcov
    if exist .coverage del /q .coverage
    echo Limpieza completada
    goto :end
)

if "%1"=="help" (
    goto :help
)

if "%1"=="" (
    goto :help
)

:help
echo Comandos disponibles:
echo.
echo   MAKE_TEST_COMMANDS.bat all          - Ejecutar todos los tests
echo   MAKE_TEST_COMMANDS.bat coverage     - Tests con cobertura
echo   MAKE_TEST_COMMANDS.bat fast         - Tests rapidos (sin slow)
echo   MAKE_TEST_COMMANDS.bat unit         - Solo tests unitarios
echo   MAKE_TEST_COMMANDS.bat integration  - Solo tests de integracion
echo   MAKE_TEST_COMMANDS.bat api          - Solo tests de API
echo   MAKE_TEST_COMMANDS.bat parallel     - Tests en paralelo
echo   MAKE_TEST_COMMANDS.bat failed       - Solo tests fallidos
echo   MAKE_TEST_COMMANDS.bat clean        - Limpiar cache
echo   MAKE_TEST_COMMANDS.bat help         - Mostrar esta ayuda
goto :end

:end

