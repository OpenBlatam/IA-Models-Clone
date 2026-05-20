#!/bin/bash

# Script de inicio para Artist Manager AI

set -e

echo "🚀 Starting Artist Manager AI..."

# Verificar variables de entorno
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  WARNING: OPENROUTER_API_KEY not set"
fi

# Crear directorios necesarios
mkdir -p data backups exports logs

# Iniciar aplicación
if [ "$1" == "dev" ]; then
    echo "📦 Starting in development mode..."
    python -m uvicorn api.app_factory:app --host 0.0.0.0 --port 8000 --reload
else
    echo "📦 Starting in production mode..."
    python -m uvicorn api.app_factory:app --host 0.0.0.0 --port 8000
fi

