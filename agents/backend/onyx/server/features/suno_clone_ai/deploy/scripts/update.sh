#!/bin/bash
# Update script for Suno Clone AI
# Pulls latest code and updates deployment

set -euo pipefail

# Configuration
readonly PROJECT_DIR="${HOME}/suno-clone-ai"
readonly CONTAINER_NAME="suno-clone-ai"
readonly IMAGE_NAME="suno-clone-ai"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Pull latest code
pull_code() {
    log_info "Pulling latest code..."
    
    if [ -d "${PROJECT_DIR}/.git" ]; then
        cd "${PROJECT_DIR}"
        git pull origin main || {
            log_error "Failed to pull latest code"
            return 1
        }
        log_info "Code updated"
    else
        log_warn "Not a git repository, skipping code pull"
    fi
}

# Update dependencies
update_dependencies() {
    log_info "Updating dependencies..."
    
    if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
        docker run --rm \
            -v "${PROJECT_DIR}:/app" \
            -w /app \
            python:3.11-slim \
            pip install --upgrade -r requirements.txt || {
            log_warn "Failed to update dependencies"
        }
    fi
}

# Rebuild Docker image
rebuild_image() {
    log_info "Rebuilding Docker image..."
    
    cd "${PROJECT_DIR}"
    docker build -t "${IMAGE_NAME}:latest" . || {
        log_error "Failed to rebuild image"
        return 1
    }
    
    # Tag with timestamp
    local timestamp=$(date +%Y%m%d_%H%M%S)
    docker tag "${IMAGE_NAME}:latest" "${IMAGE_NAME}:${timestamp}"
    
    log_info "Image rebuilt and tagged as ${IMAGE_NAME}:${timestamp}"
}

# Restart container
restart_container() {
    log_info "Restarting container..."
    
    docker restart "${CONTAINER_NAME}" || {
        log_error "Failed to restart container"
        return 1
    }
    
    # Wait for health check
    log_info "Waiting for health check..."
    sleep 30
    
    if curl -f http://localhost:8020/health &> /dev/null; then
        log_info "✅ Update successful!"
    else
        log_error "Health check failed after update"
        docker logs "${CONTAINER_NAME}" --tail 50
        return 1
    fi
}

# Main function
main() {
    log_info "Starting update process..."
    
    pull_code
    update_dependencies
    rebuild_image
    restart_container
    
    log_info "✅ Update completed successfully!"
}

main "$@"




