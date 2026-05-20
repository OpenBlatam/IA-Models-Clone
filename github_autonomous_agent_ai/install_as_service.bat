@echo off
REM Script para instalar el GitHub Autonomous Agent AI como servicio de Windows
REM Esto permite que el agente se inicie automaticamente al arrancar la computadora

echo ========================================
echo Instalacion como Servicio de Windows
echo GitHub Autonomous Agent AI
echo ========================================
echo.
echo Este script instalara el agente como servicio de Windows
echo El agente se iniciara automaticamente al arrancar la computadora
echo.
echo IMPORTANTE: Necesitas ejecutar esto como Administrador
echo.

REM Verificar si se ejecuta como administrador
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Este script debe ejecutarse como Administrador
    echo Haz clic derecho y selecciona "Ejecutar como administrador"
    pause
    exit /b 1
)

REM Cambiar al directorio del proyecto
cd /d "%~dp0"

REM Crear script de inicio para el servicio
echo Creando script de inicio...

(
echo @echo off
echo cd /d "%~dp0"
echo if exist "venv\Scripts\activate.bat" ^(
echo     call venv\Scripts\activate.bat
echo ^)
echo python -m github_autonomous_agent_ai.main --mode service
) > start_service_internal.bat

echo.
echo Para instalar como servicio de Windows, puedes usar NSSM (Non-Sucking Service Manager)
echo o crear una tarea programada en el Programador de tareas de Windows.
echo.
echo Opcion 1: Usar NSSM (recomendado)
echo   1. Descarga NSSM desde: https://nssm.cc/download
echo   2. Extrae nssm.exe en una carpeta (ej: C:\nssm)
echo   3. Ejecuta como administrador:
echo      C:\nssm\nssm.exe install GitHubAutonomousAgentAI "%~dp0start_service_internal.bat"
echo      C:\nssm\nssm.exe set GitHubAutonomousAgentAI AppDirectory "%~dp0"
echo      C:\nssm\nssm.exe set GitHubAutonomousAgentAI Description "GitHub Autonomous Agent AI - Servicio persistente"
echo      C:\nssm\nssm.exe set GitHubAutonomousAgentAI Start SERVICE_AUTO_START
echo.
echo Opcion 2: Usar Tarea Programada
echo   1. Abre el Programador de tareas de Windows
echo   2. Crea una tarea nueva
echo   3. Configura para ejecutarse "Al iniciar sesion" o "Al iniciar el equipo"
echo   4. Accion: Iniciar un programa
echo   5. Programa: "%~dp0start_service_internal.bat"
echo   6. Marca "Ejecutar con los privilegios mas altos"
echo.

pause

