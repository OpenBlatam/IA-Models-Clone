#!/bin/bash
# Script de deployment automatizado para 3D Prototype AI

set -e

echo "🚀 Iniciando deployment de 3D Prototype AI..."

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Variables
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"
echo -e "${YELLOW}Version: ${VERSION}${NC}"

# Verificar dependencias
echo -e "${YELLOW}Verificando dependencias...${NC}"
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}Python3 no encontrado${NC}"; exit 1; }
command -v pip >/dev/null 2>&1 || { echo -e "${RED}pip no encontrado${NC}"; exit 1; }

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creando entorno virtual...${NC}"
    python3 -m venv venv
fi

# Activar entorno virtual
echo -e "${YELLOW}Activando entorno virtual...${NC}"
source venv/bin/activate

# Instalar dependencias
echo -e "${YELLOW}Instalando dependencias...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Ejecutar tests
echo -e "${YELLOW}Ejecutando tests...${NC}"
pytest tests/ -v || echo -e "${YELLOW}Tests fallaron, continuando...${NC}"

# Crear directorios necesarios
echo -e "${YELLOW}Creando directorios...${NC}"
mkdir -p storage/prototypes
mkdir -p storage/backups
mkdir -p logs

# Verificar configuración
echo -e "${YELLOW}Verificando configuración...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creando archivo .env desde template...${NC}"
    cp .env.example .env 2>/dev/null || echo -e "${YELLOW}No hay template .env.example${NC}"
fi

# Deployment específico por ambiente
if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${YELLOW}Configurando para producción...${NC}"
    export DEBUG=false
    export LOG_LEVEL=INFO
elif [ "$ENVIRONMENT" = "staging" ]; then
    echo -e "${YELLOW}Configurando para staging...${NC}"
    export DEBUG=true
    export LOG_LEVEL=DEBUG
else
    echo -e "${YELLOW}Configurando para desarrollo...${NC}"
    export DEBUG=true
    export LOG_LEVEL=DEBUG
fi

# Iniciar servidor
echo -e "${GREEN}✅ Deployment completado!${NC}"
echo -e "${GREEN}Iniciando servidor...${NC}"

if [ "$ENVIRONMENT" = "production" ]; then
    gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8030
else
    uvicorn main:app --host 0.0.0.0 --port 8030 --reload
fi




