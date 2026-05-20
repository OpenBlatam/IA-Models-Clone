#!/bin/bash
# Docker run script with proper configuration

set -e

ENVIRONMENT=${1:-development}
COMPOSE_FILE="docker-compose.yml"

# Select compose file based on environment
if [ "$ENVIRONMENT" == "production" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    echo "Starting production environment..."
elif [ "$ENVIRONMENT" == "development" ]; then
    COMPOSE_FILE="docker-compose.dev.yml"
    echo "Starting development environment..."
else
    echo "Starting default environment..."
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating template..."
    cat > .env << EOF
ENVIRONMENT=$ENVIRONMENT
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
LOG_LEVEL=INFO
CACHE_ENABLED=true
REDIS_PASSWORD=changeme
POSTGRES_PASSWORD=changeme
DATABASE_URL=postgresql://music_analyzer:changeme@postgres:5432/music_analyzer_db
GRAFANA_PASSWORD=admin
EOF
    echo "Created .env template. Please update with your values."
fi

# Run docker-compose
cd "$(dirname "$0")/.."
docker-compose -f deployment/$COMPOSE_FILE --env-file .env up -d

echo "Services started!"
echo ""
echo "Available services:"
echo "  - API: http://localhost:8010"
echo "  - Health: http://localhost:8010/health"
if [ "$ENVIRONMENT" == "development" ] || [ "$ENVIRONMENT" == "production" ]; then
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
fi
echo ""
echo "View logs: docker-compose -f deployment/$COMPOSE_FILE logs -f"
echo "Stop services: docker-compose -f deployment/$COMPOSE_FILE down"




