#!/bin/bash
# Optimization script - Optimize Docker images and system
# Usage: ./scripts/optimize.sh [--images] [--system] [--all]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

OPTIMIZE_IMAGES=false
OPTIMIZE_SYSTEM=false

if [ "$1" == "--all" ] || [ "$1" == "--images" ]; then
    OPTIMIZE_IMAGES=true
fi

if [ "$1" == "--all" ] || [ "$1" == "--system" ]; then
    OPTIMIZE_SYSTEM=true
fi

if [ "$1" == "--all" ] || [ -z "$1" ]; then
    OPTIMIZE_IMAGES=true
    OPTIMIZE_SYSTEM=true
fi

echo -e "${BLUE}⚡ Optimizing Manuales Hogar AI${NC}"
echo ""

# Optimize Docker images
if [ "$OPTIMIZE_IMAGES" = true ]; then
    echo -e "${BLUE}=== Optimizing Docker Images ===${NC}"
    
    # Remove unused images
    echo "Removing unused images..."
    docker image prune -f
    echo -e "${GREEN}✅ Unused images removed${NC}"
    
    # Remove dangling images
    echo "Removing dangling images..."
    docker image prune -a -f
    echo -e "${GREEN}✅ Dangling images removed${NC}"
    
    # Clean build cache
    echo "Cleaning build cache..."
    docker builder prune -f
    echo -e "${GREEN}✅ Build cache cleaned${NC}"
    
    echo ""
fi

# Optimize system
if [ "$OPTIMIZE_SYSTEM" = true ]; then
    echo -e "${BLUE}=== Optimizing System ===${NC}"
    
    # Remove stopped containers
    echo "Removing stopped containers..."
    docker container prune -f
    echo -e "${GREEN}✅ Stopped containers removed${NC}"
    
    # Remove unused volumes
    echo "Removing unused volumes..."
    docker volume prune -f
    echo -e "${GREEN}✅ Unused volumes removed${NC}"
    
    # Remove unused networks
    echo "Removing unused networks..."
    docker network prune -f
    echo -e "${GREEN}✅ Unused networks removed${NC}"
    
    echo ""
fi

# Show space saved
echo -e "${BLUE}=== Disk Usage ===${NC}"
docker system df
echo ""

echo -e "${GREEN}🎉 Optimization complete!${NC}"
echo ""




