#!/bin/bash
# Build Docker images for Manuales Hogar AI

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
IMAGE_NAME="manuales-hogar-ai"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-dev}"

echo -e "${YELLOW}Building Docker image: ${IMAGE_NAME}:${VERSION} (${ENVIRONMENT})${NC}"

if [ "$ENVIRONMENT" == "prod" ]; then
    docker build -f Dockerfile.prod -t ${IMAGE_NAME}:${VERSION} -t ${IMAGE_NAME}:latest .
    echo -e "${GREEN}Production image built successfully${NC}"
elif [ "$ENVIRONMENT" == "dev" ]; then
    docker build -f Dockerfile.dev -t ${IMAGE_NAME}:dev -t ${IMAGE_NAME}:latest .
    echo -e "${GREEN}Development image built successfully${NC}"
else
    docker build -t ${IMAGE_NAME}:${VERSION} .
    echo -e "${GREEN}Image built successfully${NC}"
fi

echo -e "${GREEN}Image tags:${NC}"
docker images | grep ${IMAGE_NAME}




