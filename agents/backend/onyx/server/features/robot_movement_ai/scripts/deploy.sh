#!/bin/bash
# Script de deployment para producción - Robot Movement AI v2.0

set -e

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Iniciando deployment de Robot Movement AI v2.0...${NC}"

# Verificar que estamos en el directorio correcto
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt no encontrado${NC}"
    exit 1
fi

# Variables de entorno
ENV=${1:-production}
DOCKER_BUILD=${DOCKER_BUILD:-true}

echo -e "${BLUE}📋 Configuración:${NC}"
echo -e "   Entorno: ${ENV}"
echo -e "   Docker Build: ${DOCKER_BUILD}"

# Pre-deployment checks
echo -e "${BLUE}🔍 Ejecutando pre-deployment checks...${NC}"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 no encontrado${NC}"
    exit 1
fi

# Ejecutar tests
echo -e "${BLUE}🧪 Ejecutando tests...${NC}"
if [ -f "scripts/run_tests.sh" ]; then
    bash scripts/run_tests.sh --no-coverage || {
        echo -e "${RED}❌ Tests fallaron. Deployment cancelado.${NC}"
        exit 1
    }
else
    pytest tests/ -v || {
        echo -e "${RED}❌ Tests fallaron. Deployment cancelado.${NC}"
        exit 1
    }
fi

# Verificar archivo .env
if [ ! -f ".env" ] && [ "$ENV" = "production" ]; then
    echo -e "${YELLOW}⚠️  Archivo .env no encontrado${NC}"
    echo -e "${YELLOW}⚠️  Asegúrate de configurar las variables de entorno${NC}"
fi

# Build Docker si está habilitado
if [ "$DOCKER_BUILD" = "true" ] && [ -f "Dockerfile" ]; then
    echo -e "${BLUE}🐳 Construyendo imagen Docker...${NC}"
    docker build -t robot-movement-ai:latest .
    echo -e "${GREEN}✅ Imagen Docker construida${NC}"
fi

# Backup de base de datos si existe
if [ -f "db/robots.db" ]; then
    echo -e "${BLUE}💾 Creando backup de base de datos...${NC}"
    BACKUP_FILE="db/backup_$(date +%Y%m%d_%H%M%S).db"
    cp db/robots.db "$BACKUP_FILE"
    echo -e "${GREEN}✅ Backup creado: $BACKUP_FILE${NC}"
fi

# Deployment
echo -e "${BLUE}📦 Iniciando deployment...${NC}"

if [ "$DOCKER_BUILD" = "true" ] && command -v docker-compose &> /dev/null; then
    echo -e "${BLUE}🐳 Usando Docker Compose...${NC}"
    docker-compose up -d --build
    echo -e "${GREEN}✅ Servicios iniciados con Docker Compose${NC}"
else
    echo -e "${BLUE}🐍 Deployment directo con Python...${NC}"
    # Activar entorno virtual si existe
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Instalar/actualizar dependencias
    pip install -r requirements.txt
    
    echo -e "${GREEN}✅ Dependencias instaladas${NC}"
    echo -e "${YELLOW}⚠️  Inicia el servidor manualmente: python -m robot_movement_ai.main${NC}"
fi

# Health check
echo -e "${BLUE}🏥 Verificando health check...${NC}"
sleep 5

if command -v curl &> /dev/null; then
    PORT=${API_PORT:-8010}
    if curl -f http://localhost:${PORT}/health &> /dev/null; then
        echo -e "${GREEN}✅ Health check exitoso${NC}"
    else
        echo -e "${YELLOW}⚠️  Health check falló. Verifica los logs.${NC}"
    fi
fi

echo ""
echo -e "${GREEN}✅ Deployment completado!${NC}"
echo ""
echo -e "${BLUE}📝 Próximos pasos:${NC}"
echo -e "  1. Verifica los logs: docker-compose logs -f (si usas Docker)"
echo -e "  2. Verifica el health endpoint: curl http://localhost:${PORT:-8010}/health"
echo -e "  3. Monitorea las métricas y logs"
echo ""




