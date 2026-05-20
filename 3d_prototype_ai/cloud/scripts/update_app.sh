#!/bin/bash
# Update application on deployed EC2 instance

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load configuration
if [ -f "${CLOUD_DIR}/.env" ]; then
    source "${CLOUD_DIR}/.env"
fi

# Default values
INSTANCE_IP="${INSTANCE_IP}"
AWS_KEY_PATH="${AWS_KEY_PATH}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
if [ -z "${INSTANCE_IP}" ]; then
    log_error "INSTANCE_IP not set in .env file"
    exit 1
fi

if [ -z "${AWS_KEY_PATH}" ]; then
    log_error "AWS_KEY_PATH not set in .env file"
    exit 1
fi

log_info "Updating application on ${INSTANCE_IP}..."

# Copy updated files
log_info "Copying application files..."
rsync -avz \
    -e "ssh -i ${AWS_KEY_PATH} -o StrictHostKeyChecking=no" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='cloud/' \
    "${PROJECT_ROOT}/" \
    ubuntu@${INSTANCE_IP}:/opt/3d-prototype-ai/

# Update on remote instance
log_info "Updating application..."
ssh -i "${AWS_KEY_PATH}" \
    -o StrictHostKeyChecking=no \
    ubuntu@${INSTANCE_IP} << 'REMOTE_EOF'
cd /opt/3d-prototype-ai
sudo chown -R ubuntu:ubuntu /opt/3d-prototype-ai

# Update with Docker Compose
if [ -f "docker-compose.yml" ]; then
    docker-compose pull
    docker-compose up -d --build
    docker-compose ps
else
    # Update Python dependencies
    if [ -f "requirements.txt" ] && [ -d "venv" ]; then
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
    
    # Restart systemd service
    sudo systemctl daemon-reload
    sudo systemctl restart 3d-prototype-ai
fi

# Health check
sleep 5
curl -f http://localhost:8030/health && echo "✓ Application is healthy" || echo "✗ Health check failed"
REMOTE_EOF

log_info "Application updated successfully!"

