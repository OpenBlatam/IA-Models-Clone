#!/bin/bash
# Script de inicio para Autonomous Long-Term Agent

echo "🚀 Iniciando Autonomous Long-Term Agent..."
echo ""

# Verificar que existe .env
if [ ! -f .env ]; then
    echo "⚠️  Archivo .env no encontrado"
    echo "   Creando archivo .env de ejemplo..."
    cat > .env << EOF
# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_HTTP_REFERER=https://blatam-academy.com
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Agent
AGENT_POLL_INTERVAL=1.0
AGENT_MAX_CONCURRENT_TASKS=10
AGENT_MAX_PARALLEL_INSTANCES=5

# Learning
LEARNING_ENABLED=true
LEARNING_ADAPTATION_RATE=0.1

# Server
HOST=0.0.0.0
PORT=8001
EOF
    echo "   ✅ Archivo .env creado. Por favor configura OPENROUTER_API_KEY"
    exit 1
fi

# Verificar que OPENROUTER_API_KEY esté configurado
if ! grep -q "OPENROUTER_API_KEY=sk-or" .env 2>/dev/null; then
    echo "⚠️  OPENROUTER_API_KEY no está configurado en .env"
    echo "   Por favor configura tu API key de OpenRouter"
    exit 1
fi

# Iniciar servidor
echo "✅ Configuración verificada"
echo "📡 Iniciando servidor en http://localhost:8001"
echo "📚 Documentación disponible en http://localhost:8001/docs"
echo ""
echo "⚠️  IMPORTANTE: Los agentes NO se detienen automáticamente"
echo "   Usa el endpoint /api/v1/agents/{agent_id}/stop para detenerlos"
echo ""

python -m autonomous_long_term_agent.main




