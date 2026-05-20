@echo off
REM Script para iniciar el GitHub Autonomous Agent AI como servicio persistente
REM Este script inicia el agente y NO se detiene hasta que presiones Ctrl+C o cierres la ventana

echo ========================================
echo GitHub Autonomous Agent AI
echo Servicio Persistente
echo ========================================
echo.
echo IMPORTANTE: El agente NO se detendra automaticamente
echo Solo se detendra cuando presiones el boton de parar en la interfaz web
echo.
echo Presiona Ctrl+C para detener (solo si es necesario)
echo.

REM Cambiar al directorio del proyecto
cd /d "%~dp0"

REM Activar entorno virtual si existe
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Iniciar el servicio en modo persistente
python -m github_autonomous_agent_ai.main --mode service

pause

