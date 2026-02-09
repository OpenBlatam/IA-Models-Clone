#!/bin/bash
# Status script - Check service status
# Usage: ./status.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}📊 Manuales Hogar AI - Service Status${NC}"
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi

# Use 'docker compose' if 'docker-compose' is not available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Check which compose file to use
if [ -f docker-compose.prod.yml ] && docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    COMPOSE_FILE="-f docker-compose.prod.yml"
    ENV="production"
elif docker-compose ps | grep -q "Up"; then
    COMPOSE_FILE=""
    ENV="development"
else
    echo -e "${YELLOW}⚠️  No services are running${NC}"
    echo ""
    echo "Start services with: ./start.sh"
    exit 0
fi

echo -e "${BLUE}Environment:${NC} $ENV"
echo ""

# Show container status
echo -e "${BLUE}Container Status:${NC}"
$DOCKER_COMPOSE $COMPOSE_FILE ps
echo ""

# Check health endpoint
echo -e "${BLUE}Health Check:${NC}"
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Service is healthy${NC}"
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Service is not responding${NC}"
fi
echo ""

# Show resource usage
echo -e "${BLUE}Resource Usage:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $($DOCKER_COMPOSE $COMPOSE_FILE ps -q) 2>/dev/null || echo "Unable to get stats"
echo ""

# Show recent logs (last 5 lines)
echo -e "${BLUE}Recent Logs (last 5 lines):${NC}"
$DOCKER_COMPOSE $COMPOSE_FILE logs --tail=5 app 2>/dev/null || echo "No logs available"
echo ""




