#!/bin/bash
# Setup script - Initial setup and configuration
# Usage: ./scripts/setup.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 Setting up Manuales Hogar AI...${NC}"
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file${NC}"
        echo -e "${YELLOW}⚠️  Please edit .env and set your OPENROUTER_API_KEY${NC}"
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p checkpoints
echo -e "${GREEN}✅ Directories created${NC}"

# Check if images need to be built
echo ""
echo -e "${BLUE}Checking Docker images...${NC}"
if docker images | grep -q "manuales-hogar-ai"; then
    echo -e "${GREEN}✅ Docker images found${NC}"
else
    echo -e "${YELLOW}⚠️  Docker images not found. They will be built on first start.${NC}"
fi

# Check Python dependencies (if running locally)
if command -v python3 &> /dev/null; then
    echo ""
    echo -e "${BLUE}Checking Python environment...${NC}"
    if [ -d ".venv" ] || [ -d "venv" ]; then
        echo -e "${GREEN}✅ Python virtual environment found${NC}"
    else
        echo -e "${YELLOW}⚠️  Python virtual environment not found${NC}"
        echo -e "${BLUE}   Creating virtual environment...${NC}"
        python3 -m venv .venv
        echo -e "${GREEN}✅ Virtual environment created${NC}"
        echo -e "${BLUE}   Installing dependencies...${NC}"
        source .venv/bin/activate
        pip install -r requirements.txt
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    fi
fi

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Edit .env and set your OPENROUTER_API_KEY"
echo "2. Run ./start.sh to start the service"
echo "3. Access the API at http://localhost:8000"
echo ""




