#!/bin/bash
# Stop service script for EC2
# This script stops the Manuales Hogar AI application

set -e

APP_DIR="/opt/manuales-hogar-ai"
cd $APP_DIR

echo "Stopping Manuales Hogar AI service..."

# Stop Docker Compose
if [ -f "$APP_DIR/docker-compose.yml" ]; then
    docker compose down
elif [ -f "$APP_DIR/docker-compose.prod.yml" ]; then
    docker compose -f docker-compose.prod.yml down
fi

echo "Service stopped successfully"




