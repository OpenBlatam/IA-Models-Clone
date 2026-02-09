#!/bin/bash
# Script para ejecutar tests - Robot Movement AI v2.0

set -e

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🧪 Ejecutando tests para Robot Movement AI v2.0...${NC}"

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Opciones por defecto
COVERAGE=true
VERBOSE=false
PATTERN=""

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -k|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        *)
            echo "Uso: $0 [--no-coverage] [-v|--verbose] [-k|--pattern PATTERN]"
            exit 1
            ;;
    esac
done

# Construir comando pytest
CMD="pytest tests/"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD -v"
fi

if [ -n "$PATTERN" ]; then
    CMD="$CMD -k $PATTERN"
fi

if [ "$COVERAGE" = true ]; then
    CMD="$CMD --cov=core --cov=api --cov-report=html --cov-report=term-missing"
fi

# Ejecutar tests
echo -e "${BLUE}📊 Ejecutando: $CMD${NC}"
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Todos los tests pasaron!${NC}"
    if [ "$COVERAGE" = true ]; then
        echo -e "${BLUE}📊 Reporte de cobertura generado en: htmlcov/index.html${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}⚠️  Algunos tests fallaron${NC}"
    exit 1
fi




