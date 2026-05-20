#!/bin/bash
# Start service script for EC2
# This script starts the Manuales Hogar AI application

set -e

APP_DIR="/opt/manuales-hogar-ai"
cd $APP_DIR

echo "Starting Manuales Hogar AI service..."

# Check if .env exists
if [ ! -f "$APP_DIR/.env" ]; then
    echo "ERROR: .env file not found at $APP_DIR/.env"
    echo "Please create .env file from .env.example"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Start Docker if not running
if ! systemctl is-active --quiet docker; then
    echo "Starting Docker..."
    sudo systemctl start docker
fi

# Check if docker-compose.yml exists
if [ -f "$APP_DIR/docker-compose.yml" ]; then
    echo "Starting with Docker Compose..."
    docker compose up -d
elif [ -f "$APP_DIR/docker-compose.prod.yml" ]; then
    echo "Starting with Docker Compose (production)..."
    docker compose -f docker-compose.prod.yml up -d
else
    echo "ERROR: docker-compose.yml not found"
    exit 1
fi

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 10

# Check health
if command -v curl &> /dev/null; then
    PORT=$(grep PORT $APP_DIR/.env | cut -d '=' -f2 | tr -d ' ' || echo "8000")
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f -s http://localhost:$PORT/api/v1/health > /dev/null 2>&1; then
            echo "Service is healthy!"
            exit 0
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 2
    done
    
    echo "WARNING: Service may not be fully healthy, but started"
fi

echo "Service started successfully"




