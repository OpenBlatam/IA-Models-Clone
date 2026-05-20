#!/bin/bash

# Setup script for AI Job Replacement Helper

echo "🚀 Setting up AI Job Replacement Helper..."

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Instalar dependencias
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Crear directorios necesarios
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p exports

# Copiar .env.example si no existe .env
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Ejecutar migraciones (si hay base de datos)
# echo "🗄️  Running migrations..."
# alembic upgrade head

echo "✅ Setup complete!"
echo "📝 Next steps:"
echo "   1. Edit .env file with your configuration"
echo "   2. Run: python main.py"
echo "   3. Visit: http://localhost:8030"




