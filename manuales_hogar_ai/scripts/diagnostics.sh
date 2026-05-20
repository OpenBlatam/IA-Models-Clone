#!/bin/bash
# Diagnostics script - Comprehensive system diagnostics
# Usage: ./scripts/diagnostics.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🔍 Manuales Hogar AI - System Diagnostics${NC}"
echo "=========================================="
echo ""

# System information
echo -e "${CYAN}=== System Information ===${NC}"
echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""

# Docker information
echo -e "${CYAN}=== Docker Information ===${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker installed${NC}"
    echo "Version: $(docker --version)"
    echo ""
    
    if docker info > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Docker is running${NC}"
        echo "Containers: $(docker ps -q | wc -l) running"
        echo "Images: $(docker images -q | wc -l) available"
    else
        echo -e "${RED}❌ Docker is not running${NC}"
    fi
else
    echo -e "${RED}❌ Docker is not installed${NC}"
fi
echo ""

# Docker Compose
echo -e "${CYAN}=== Docker Compose ===${NC}"
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✅ docker-compose installed${NC}"
    echo "Version: $(docker-compose --version)"
elif docker compose version &> /dev/null; then
    echo -e "${GREEN}✅ docker compose available${NC}"
    echo "Version: $(docker compose version)"
else
    echo -e "${RED}❌ docker-compose is not available${NC}"
fi
echo ""

# Service status
echo -e "${CYAN}=== Service Status ===${NC}"
if docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo -e "${GREEN}✅ Services are running${NC}"
    docker-compose ps
else
    echo -e "${YELLOW}⚠️  Services are not running${NC}"
fi
echo ""

# Port availability
echo -e "${CYAN}=== Port Availability ===${NC}"
for port in 8000 5432 6379; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "Port $port: ${YELLOW}⚠️  In use${NC}"
    else
        echo -e "Port $port: ${GREEN}✅ Available${NC}"
    fi
done
echo ""

# Disk space
echo -e "${CYAN}=== Disk Space ===${NC}"
df -h . | tail -n 1
echo ""

# Environment file
echo -e "${CYAN}=== Environment Configuration ===${NC}"
if [ -f .env ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
    if grep -q "OPENROUTER_API_KEY" .env && ! grep -q "OPENROUTER_API_KEY=$" .env; then
        echo -e "${GREEN}✅ OPENROUTER_API_KEY is set${NC}"
    else
        echo -e "${YELLOW}⚠️  OPENROUTER_API_KEY is not set${NC}"
    fi
else
    echo -e "${RED}❌ .env file not found${NC}"
fi
echo ""

# Health check
echo -e "${CYAN}=== Health Check ===${NC}"
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is responding${NC}"
    HEALTH=$(curl -s http://localhost:8000/api/v1/health 2>/dev/null)
    echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
else
    echo -e "${RED}❌ API is not responding${NC}"
fi
echo ""

# Database connectivity
echo -e "${CYAN}=== Database Connectivity ===${NC}"
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Database is accessible${NC}"
    DB_VERSION=$(docker-compose exec -T postgres psql -U postgres -t -c "SELECT version();" 2>/dev/null | head -n 1)
    echo "Version: $DB_VERSION"
else
    echo -e "${RED}❌ Database is not accessible${NC}"
fi
echo ""

# Recent errors
echo -e "${CYAN}=== Recent Errors (last 10 lines) ===${NC}"
docker-compose logs --tail=10 2>/dev/null | grep -i error || echo "No errors found"
echo ""

# Summary
echo -e "${CYAN}=== Summary ===${NC}"
ISSUES=0

if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    ISSUES=$((ISSUES + 1))
fi

if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file missing${NC}"
    ISSUES=$((ISSUES + 1))
fi

if ! curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  API is not responding${NC}"
    ISSUES=$((ISSUES + 1))
fi

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
else
    echo -e "${YELLOW}⚠️  Found $ISSUES issue(s)${NC}"
fi
echo ""




