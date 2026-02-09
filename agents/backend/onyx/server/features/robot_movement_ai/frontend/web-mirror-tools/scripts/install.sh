#!/bin/bash
# install.sh - Script de instalación de dependencias

set -e

echo "🚀 Instalando dependencias para Web Mirror Tools"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    echo "   Instala Python 3.8 o superior desde https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python encontrado: $(python3 --version)"

# Instalar dependencias Python
echo ""
echo "📦 Instalando dependencias Python..."
cd "$(dirname "$0")/.."
python3 -m pip install --upgrade pip
python3 -m pip install -r config/requirements.txt

echo ""
echo "✅ Dependencias Python instaladas"

# Preguntar si instalar Playwright
echo ""
read -p "¿Instalar Playwright para contenido JavaScript? (s/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[SsYy]$ ]]; then
    echo "📦 Instalando Playwright..."
    python3 -m pip install playwright
    python3 -m playwright install chromium
    echo "✅ Playwright instalado"
fi

# Verificar wget
echo ""
if command -v wget &> /dev/null; then
    echo "✅ wget encontrado: $(wget --version | head -n1)"
else
    echo "⚠️  wget no está instalado (opcional)"
    echo "   Para instalarlo:"
    echo "   - Ubuntu/Debian: sudo apt-get install wget"
    echo "   - macOS: brew install wget"
    echo "   - Windows: choco install wget"
fi

# Verificar httrack
echo ""
if command -v httrack &> /dev/null; then
    echo "✅ httrack encontrado: $(httrack --version | head -n1)"
else
    echo "⚠️  httrack no está instalado (opcional)"
    echo "   Para instalarlo:"
    echo "   - Ubuntu/Debian: sudo apt-get install httrack"
    echo "   - macOS: brew install httrack"
    echo "   - Windows: https://www.httrack.com/"
fi

echo ""
echo "🎉 Instalación completada!"
echo ""
echo "Próximos pasos:"
echo "  1. Lee LEGAL_CHECKLIST.md"
echo "  2. Ejecuta: python scripts/validate_legal.py --url <URL>"
echo "  3. Ver QUICK_START.md para ejemplos de uso"



