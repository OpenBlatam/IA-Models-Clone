#!/bin/bash
# Validate configuration script
# Usage: ./scripts/validate-config.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo -e "${BLUE}✅ Validating Configuration${NC}"
echo "=============================="
echo ""

# Check .env file
echo -e "${BLUE}=== Environment File ===${NC}"
if [ -f .env ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
    
    # Check required variables
    REQUIRED_VARS=("OPENROUTER_API_KEY")
    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${var}=" .env && ! grep -q "^${var}=$" .env; then
            echo -e "${GREEN}✅ $var is set${NC}"
        else
            echo -e "${RED}❌ $var is not set${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    done
    
    # Check optional but recommended variables
    OPTIONAL_VARS=("DB_HOST" "DB_PORT" "DB_USER" "DB_PASSWORD" "DB_NAME")
    for var in "${OPTIONAL_VARS[@]}"; do
        if grep -q "^${var}=" .env; then
            echo -e "${GREEN}✅ $var is set${NC}"
        else
            echo -e "${YELLOW}⚠️  $var is not set (optional)${NC}"
            WARNINGS=$((WARNINGS + 1))
        fi
    done
else
    echo -e "${RED}❌ .env file not found${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check docker-compose files
echo -e "${BLUE}=== Docker Compose Files ===${NC}"
if [ -f docker-compose.yml ]; then
    echo -e "${GREEN}✅ docker-compose.yml exists${NC}"
    
    # Validate YAML syntax
    if command -v docker-compose &> /dev/null; then
        if docker-compose config > /dev/null 2>&1; then
            echo -e "${GREEN}✅ docker-compose.yml is valid${NC}"
        else
            echo -e "${RED}❌ docker-compose.yml has errors${NC}"
            docker-compose config
            ERRORS=$((ERRORS + 1))
        fi
    fi
else
    echo -e "${RED}❌ docker-compose.yml not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ -f docker-compose.prod.yml ]; then
    echo -e "${GREEN}✅ docker-compose.prod.yml exists${NC}"
else
    echo -e "${YELLOW}⚠️  docker-compose.prod.yml not found (optional)${NC}"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check Dockerfiles
echo -e "${BLUE}=== Dockerfiles ===${NC}"
if [ -f Dockerfile ]; then
    echo -e "${GREEN}✅ Dockerfile exists${NC}"
else
    echo -e "${RED}❌ Dockerfile not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ -f Dockerfile.dev ]; then
    echo -e "${GREEN}✅ Dockerfile.dev exists${NC}"
fi

if [ -f Dockerfile.prod ]; then
    echo -e "${GREEN}✅ Dockerfile.prod exists${NC}"
fi
echo ""

# Check required directories
echo -e "${BLUE}=== Directories ===${NC}"
REQUIRED_DIRS=("api" "core" "config" "database" "infrastructure")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✅ $dir/ exists${NC}"
    else
        echo -e "${RED}❌ $dir/ not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check required files
echo -e "${BLUE}=== Required Files ===${NC}"
REQUIRED_FILES=("main.py" "requirements.txt" "alembic.ini")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file exists${NC}"
    else
        echo -e "${RED}❌ $file not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ Configuration is valid!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Configuration is valid with $WARNINGS warning(s)${NC}"
    exit 0
else
    echo -e "${RED}❌ Configuration has $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    exit 1
fi




