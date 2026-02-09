#!/bin/bash
# Stop script for Music Analyzer AI
# Usage: ./stop.sh [dev|prod]

set -e

ENVIRONMENT=${1:-dev}
COMPOSE_FILE="docker-compose.yml"

# Select compose file
case $ENVIRONMENT in
  dev|development)
    COMPOSE_FILE="docker-compose.dev.yml"
    ;;
  prod|production)
    COMPOSE_FILE="docker-compose.prod.yml"
    ;;
esac

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Navigate to deployment directory
cd "$(dirname "$0")"

echo "🛑 Stopping Music Analyzer AI services..."

$DOCKER_COMPOSE -f $COMPOSE_FILE down

echo "✅ Services stopped!"




