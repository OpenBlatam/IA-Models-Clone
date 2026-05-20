#!/bin/bash
# Update script - Update application and dependencies
# Usage: ./scripts/update.sh [--pull] [--rebuild]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PULL_CODE=false
REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --pull)
            PULL_CODE=true
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🔄 Updating Manuales Hogar AI...${NC}"
echo ""

# Pull latest code
if [ "$PULL_CODE" = true ]; then
    echo -e "${BLUE}Pulling latest code...${NC}"
    if [ -d .git ]; then
        git pull || echo -e "${YELLOW}⚠️  Git pull failed or not a git repository${NC}"
    else
        echo -e "${YELLOW}⚠️  Not a git repository, skipping pull${NC}"
    fi
    echo -e "${GREEN}✅ Code updated${NC}"
    echo ""
fi

# Update requirements
echo -e "${BLUE}Checking for dependency updates...${NC}"
if [ -f requirements.txt ]; then
    echo "Current requirements.txt found"
    # Could add pip-tools or similar here for dependency checking
fi
echo ""

# Rebuild images
if [ "$REBUILD" = true ]; then
    echo -e "${BLUE}Rebuilding Docker images...${NC}"
    docker-compose build --no-cache
    echo -e "${GREEN}✅ Images rebuilt${NC}"
    echo ""
fi

# Update running containers
echo -e "${BLUE}Updating running containers...${NC}"
if docker-compose ps | grep -q "Up"; then
    echo "Stopping services..."
    docker-compose down
    
    if [ "$REBUILD" = true ]; then
        echo "Starting with rebuilt images..."
        docker-compose up -d --build
    else
        echo "Starting services..."
        docker-compose up -d
    fi
    
    echo -e "${GREEN}✅ Services updated${NC}"
else
    echo -e "${YELLOW}⚠️  Services are not running${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Update complete!${NC}"
echo ""
echo "Next steps:"
echo "  - Check status: ./status.sh"
echo "  - View logs: docker-compose logs -f"
echo "  - Test API: ./scripts/test-api.sh"




