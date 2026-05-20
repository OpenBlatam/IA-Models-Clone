#!/bin/bash
# ULTRA OPTIMIZED RUN SCRIPT
# Script de ejecución ultra optimizado

echo "🚀 Starting ULTRA OPTIMIZED AI History Comparison System..."
echo "⚡ Maximum Performance Mode Activated"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Verificar dependencias ultra
echo "🔍 Checking ultra dependencies..."
python3 -c "import fastapi, uvicorn, loguru" 2>/dev/null || {
    echo "📦 Installing ultra dependencies..."
    pip3 install fastapi uvicorn[standard] loguru
}

# Verificar dependencias de performance
python3 -c "import uvloop, httptools" 2>/dev/null || {
    echo "⚡ Installing performance boosters..."
    pip3 install uvloop httptools
}

# Verificar archivo principal
if [ ! -f "ULTRA_OPTIMIZED.py" ]; then
    echo "❌ ULTRA_OPTIMIZED.py not found!"
    exit 1
fi

echo "✅ All dependencies ready"
echo "🚀 Starting ultra optimized server..."

# Configuración ultra optimizada
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export UVICORN_WORKERS=8
export UVICORN_LOOP=uvloop
export UVICORN_HTTP=httptools

# Ejecutar ultra optimizado
uvicorn ULTRA_OPTIMIZED:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 8 \
    --loop uvloop \
    --http httptools \
    --log-level warning \
    --no-access-log \
    --no-server-header \
    --no-date-header \
    --lifespan off

echo "🛑 Ultra optimized server stopped"







