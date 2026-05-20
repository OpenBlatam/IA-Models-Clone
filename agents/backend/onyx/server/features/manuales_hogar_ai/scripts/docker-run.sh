#!/bin/bash
# Run Docker container for Manuales Hogar AI

set -e

# Configuration
IMAGE_NAME="manuales-hogar-ai"
CONTAINER_NAME="manuales-hogar-ai"
PORT="${PORT:-8000}"

# Check if container already exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME} || true
    docker rm ${CONTAINER_NAME} || true
fi

# Run container
echo "Starting container ${CONTAINER_NAME}..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8000 \
    -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
    -e DB_HOST="${DB_HOST:-localhost}" \
    -e DB_PORT="${DB_PORT:-5432}" \
    -e DB_USER="${DB_USER:-postgres}" \
    -e DB_PASSWORD="${DB_PASSWORD:-postgres}" \
    -e DB_NAME="${DB_NAME:-manuales_hogar}" \
    -e DATABASE_URL="${DATABASE_URL}" \
    ${IMAGE_NAME}:latest

echo "Container started. Access the API at http://localhost:${PORT}"
echo "View logs: docker logs -f ${CONTAINER_NAME}"
echo "Stop container: docker stop ${CONTAINER_NAME}"




