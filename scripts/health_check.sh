#!/bin/bash
# Health Check Completo del Sistema

set -e

echo "🏥 Health Check del Sistema Blatam Academy"
echo "=========================================="

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_service() {
    local service=$1
    local port=$2
    local url="http://localhost:${port}/health"
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${service} (puerto ${port}) - Healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ ${service} (puerto ${port}) - Unhealthy${NC}"
        return 1
    fi
}

check_docker() {
    if docker ps | grep -q "$1"; then
        echo -e "${GREEN}✅ Docker container $1 - Running${NC}"
    else
        echo -e "${RED}❌ Docker container $1 - Not running${NC}"
    fi
}

echo ""
echo "🔍 Verificando servicios..."

# Verificar Docker containers
check_docker "integration-system"
check_docker "bul"
check_docker "content-redundancy"
check_docker "postgres"
check_docker "redis"
check_docker "nginx"

echo ""
echo "🌐 Verificando endpoints HTTP..."

# Verificar servicios HTTP
check_service "Integration System" 8000
check_service "BUL" 8002
check_service "Content Redundancy" 8001
check_service "Business Agents" 8004
check_service "Export IA" 8005

echo ""
echo "💾 Verificando recursos del sistema..."
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo ""
echo "📊 Health check completo"

