#!/bin/bash
# Script de testing automatizado para 3D Prototype AI

set -e

echo "🧪 Ejecutando tests de 3D Prototype AI..."

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Variables
COVERAGE=${1:-false}
VERBOSE=${2:-true}

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Instalar dependencias de testing si no están
pip install -q pytest pytest-asyncio pytest-cov pytest-mock

# Ejecutar tests
echo -e "${YELLOW}Ejecutando tests...${NC}"

if [ "$COVERAGE" = "true" ]; then
    echo -e "${YELLOW}Con coverage...${NC}"
    pytest tests/ -v --cov=. --cov-report=html --cov-report=term
    echo -e "${GREEN}✅ Coverage report generado en htmlcov/index.html${NC}"
else
    if [ "$VERBOSE" = "true" ]; then
        pytest tests/ -v
    else
        pytest tests/
    fi
fi

# Verificar linting
echo -e "${YELLOW}Verificando linting...${NC}"
if command -v flake8 >/dev/null 2>&1; then
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
fi

# Verificar tipos
echo -e "${YELLOW}Verificando tipos...${NC}"
if command -v mypy >/dev/null 2>&1; then
    mypy . --ignore-missing-imports || true
fi

echo -e "${GREEN}✅ Tests completados!${NC}"




