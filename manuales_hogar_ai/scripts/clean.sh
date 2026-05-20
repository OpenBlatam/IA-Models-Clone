#!/bin/bash
# Clean script - Remove containers, volumes, and images
# Usage: ./scripts/clean.sh [--all]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CLEAN_ALL=false

if [ "$1" == "--all" ]; then
    CLEAN_ALL=true
fi

echo -e "${YELLOW}🧹 Cleaning Manuales Hogar AI...${NC}"
echo ""

# Stop and remove containers
echo -e "${BLUE}Stopping containers...${NC}"
docker-compose down -v 2>/dev/null || true
docker-compose -f docker-compose.prod.yml down -v 2>/dev/null || true
echo -e "${GREEN}✅ Containers stopped${NC}"

if [ "$CLEAN_ALL" = true ]; then
    echo ""
    echo -e "${BLUE}Removing images...${NC}"
    docker images | grep "manuales-hogar-ai" | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true
    echo -e "${GREEN}✅ Images removed${NC}"
    
    echo ""
    echo -e "${BLUE}Removing volumes...${NC}"
    docker volume ls | grep "manuales" | awk '{print $2}' | xargs docker volume rm 2>/dev/null || true
    echo -e "${GREEN}✅ Volumes removed${NC}"
    
    echo ""
    echo -e "${BLUE}Cleaning build cache...${NC}"
    docker builder prune -f
    echo -e "${GREEN}✅ Build cache cleaned${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Cleanup complete!${NC}"
echo ""




