@echo off
REM Script de inicio para Autonomous Long-Term Agent (Windows)

echo 🚀 Iniciando Autonomous Long-Term Agent...
echo.

REM Verificar que existe .env
if not exist .env (
    echo ⚠️  Archivo .env no encontrado
    echo    Creando archivo .env de ejemplo...
    (
        echo # OpenRouter
        echo OPENROUTER_API_KEY=sk-or-v1-...
        echo OPENROUTER_HTTP_REFERER=https://blatam-academy.com
        echo OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
        echo.
        echo # Agent
        echo AGENT_POLL_INTERVAL=1.0
        echo AGENT_MAX_CONCURRENT_TASKS=10
        echo AGENT_MAX_PARALLEL_INSTANCES=5
        echo.
        echo # Learning
        echo LEARNING_ENABLED=true
        echo LEARNING_ADAPTATION_RATE=0.1
        echo.
        echo # Server
        echo HOST=0.0.0.0
        echo PORT=8001
    ) > .env
    echo    ✅ Archivo .env creado. Por favor configura OPENROUTER_API_KEY
    pause
    exit /b 1
)

echo ✅ Configuración verificada
echo 📡 Iniciando servidor en http://localhost:8001
echo 📚 Documentación disponible en http://localhost:8001/docs
echo.
echo ⚠️  IMPORTANTE: Los agentes NO se detienen automáticamente
echo    Usa el endpoint /api/v1/agents/{agent_id}/stop para detenerlos
echo.

python -m autonomous_long_term_agent.main

pause




