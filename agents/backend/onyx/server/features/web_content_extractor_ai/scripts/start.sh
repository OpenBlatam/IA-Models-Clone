#!/bin/bash

# Script de inicio para Web Content Extractor AI

set -e

echo "🚀 Iniciando Web Content Extractor AI..."

# Verificar variables de entorno
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  ADVERTENCIA: OPENROUTER_API_KEY no está configurada"
    echo "   Crea un archivo .env con tu API key"
fi

# Instalar dependencias si es necesario
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python -m venv venv
fi

source venv/bin/activate

echo "📥 Instalando dependencias..."
pip install -q -r requirements.txt

echo "✅ Iniciando servidor..."
python main.py








