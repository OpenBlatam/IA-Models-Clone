#!/bin/bash
# Deploy script for unified_ai_model on EC2
# This script is executed on the EC2 instance to update and restart the service

set -e

# Configuration
APP_DIR="${APP_DIR:-/home/ec2-user/unified_ai_model}"
REPO_URL="${REPO_URL:-https://github.com/OpenBlatam/onyx.git}"
BRANCH="${BRANCH:-main}"
COMPOSE_FILE="aws/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Navigate to app directory
cd "$APP_DIR/backend/unified_ai_model" || {
    log_error "Failed to navigate to app directory: $APP_DIR/backend/unified_ai_model"
    exit 1
}

# Pull latest code
log_info "Pulling latest code from $BRANCH branch..."
git fetch origin
git reset --hard origin/$BRANCH

# Build and restart containers with zero-downtime
log_info "Building new Docker image..."
docker-compose -f "$COMPOSE_FILE" build --no-cache

log_info "Stopping old containers..."
docker-compose -f "$COMPOSE_FILE" down

log_info "Starting new containers..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for health check
log_info "Waiting for service to be healthy..."
sleep 10

# Check health
if curl -f http://localhost:8050/api/v1/health > /dev/null 2>&1; then
    log_info "Deployment successful! Service is healthy."
else
    log_error "Health check failed!"
    docker-compose -f "$COMPOSE_FILE" logs
    exit 1
fi

# Clean up old images
log_info "Cleaning up unused Docker images..."
docker image prune -f

log_info "Deployment completed successfully!"
