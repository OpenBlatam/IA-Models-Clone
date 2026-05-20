#!/bin/bash
# Bulk Chat - Linux/Mac Start Script
# ===================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "   Bulk Chat - Sistema de Chat Continuo"
echo "========================================"
echo ""

# Cambiar al directorio del script
cd "$(dirname "$0")"

# Verificar Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python no encontrado. Por favor instala Python 3.8+"
    exit 1
fi

# Usar python3 si está disponible, sino python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Verificar si estamos en el directorio correcto
if [ ! -f "main.py" ]; then
    echo -e "${RED}[ERROR]${NC} No se encuentra main.py. Asegúrate de estar en el directorio correcto."
    exit 1
fi

# Verificar dependencias básicas
if ! $PYTHON_CMD -c "import fastapi" &> /dev/null; then
    echo -e "${YELLOW}[ADVERTENCIA]${NC} FastAPI no encontrado. Instalando dependencias..."
    $PYTHON_CMD install.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Error al instalar dependencias."
        exit 1
    fi
fi

# Leer argumentos
LLM_PROVIDER=${1:-mock}
PORT=${2:-8006}
HOST=${3:-0.0.0.0}

# Mostrar ayuda si se solicita
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo ""
    echo "Uso: ./start.sh [provider] [port] [host]"
    echo ""
    echo "Opciones:"
    echo "  provider  - Proveedor LLM (openai, anthropic, mock) [default: mock]"
    echo "  port      - Puerto del servidor [default: 8006]"
    echo "  host      - Host del servidor [default: 0.0.0.0]"
    echo ""
    echo "Ejemplos:"
    echo "  ./start.sh                    - Inicia con modo mock en puerto 8006"
    echo "  ./start.sh openai 8006         - Inicia con OpenAI en puerto 8006"
    echo "  ./start.sh mock 9000           - Inicia con mock en puerto 9000"
    echo "  ./start.sh openai 8006 127.0.0.1 - Inicia con OpenAI en localhost:8006"
    echo ""
    exit 0
fi

echo ""
echo -e "${GREEN}Iniciando servidor...${NC}"
echo -e "Proveedor LLM: ${BLUE}$LLM_PROVIDER${NC}"
echo -e "Puerto: ${BLUE}$PORT${NC}"
echo -e "Host: ${BLUE}$HOST${NC}"
echo ""
echo "Para detener el servidor, presiona Ctrl+C"
echo ""

# Iniciar servidor
$PYTHON_CMD -m bulk_chat.main --llm-provider "$LLM_PROVIDER" --port "$PORT" --host "$HOST"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR]${NC} Error al iniciar el servidor."
    exit 1
fi
















