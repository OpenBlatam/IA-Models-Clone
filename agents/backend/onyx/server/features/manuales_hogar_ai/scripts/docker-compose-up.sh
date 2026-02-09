#!/bin/bash
# Start all services with docker-compose

set -e

ENVIRONMENT="${1:-dev}"

if [ "$ENVIRONMENT" == "prod" ]; then
    echo "Starting production environment..."
    docker-compose -f docker-compose.prod.yml up -d
else
    echo "Starting development environment..."
    docker-compose up -d
fi

echo "Services started. Waiting for health checks..."
sleep 10

# Show status
docker-compose ps

echo ""
echo "API available at: http://localhost:8000"
echo "View logs: docker-compose logs -f"
echo "Stop services: docker-compose down"




