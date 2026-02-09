#!/bin/bash
# Rollback script for Suno Clone AI
# Rolls back to previous Docker image version

set -euo pipefail

# Configuration
readonly CONTAINER_NAME="suno-clone-ai"
readonly IMAGE_NAME="suno-clone-ai"
readonly APP_PORT=8020
readonly HEALTH_CHECK_URL="http://localhost:${APP_PORT}/health"

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

# List available image versions
list_versions() {
    log_info "Available image versions:"
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}"
}

# Get previous image tag
get_previous_image() {
    local current_tag="${1:-latest}"
    
    # Get all tags except current
    local previous_tags=$(docker images "${IMAGE_NAME}" --format "{{.Tag}}" | grep -v "^${current_tag}$" | head -1)
    
    if [ -z "${previous_tags}" ]; then
        log_error "No previous image version found"
        return 1
    fi
    
    echo "${previous_tags}"
}

# Create backup before rollback
create_backup() {
    log_info "Creating backup before rollback..."
    
    if [ -f "${HOME}/suno-clone-ai/deploy/scripts/backup.sh" ]; then
        "${HOME}/suno-clone-ai/deploy/scripts/backup.sh" || {
            log_warn "Backup failed, continuing with rollback..."
        }
    else
        log_warn "Backup script not found, skipping backup"
    fi
}

# Rollback to previous version
rollback() {
    local target_tag="${1:-}"
    
    if [ -z "${target_tag}" ]; then
        # Get previous version automatically
        target_tag=$(get_previous_image)
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
    
    log_info "Rolling back to version: ${target_tag}"
    
    # Stop current container
    log_info "Stopping current container..."
    docker stop "${CONTAINER_NAME}" || true
    docker rm "${CONTAINER_NAME}" || true
    
    # Start with previous image
    log_info "Starting container with previous image..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        -p "${APP_PORT}:${APP_PORT}" \
        --env-file "${HOME}/suno-clone-ai/.env" \
        --health-cmd="curl -f ${HEALTH_CHECK_URL} || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        "${IMAGE_NAME}:${target_tag}" || {
        log_error "Failed to start container with previous version"
        exit 1
    }
    
    # Wait for health check
    log_info "Waiting for health check..."
    sleep 30
    
    if curl -f "${HEALTH_CHECK_URL}" &> /dev/null; then
        log_info "✅ Rollback successful!"
        log_info "Current version: ${target_tag}"
    else
        log_error "Health check failed after rollback"
        docker logs "${CONTAINER_NAME}" --tail 50
        exit 1
    fi
}

# Main function
main() {
    if [ "${1:-}" == "--list" ] || [ "${1:-}" == "-l" ]; then
        list_versions
        exit 0
    fi
    
    local target_version="${1:-}"
    
    create_backup
    rollback "${target_version}"
    
    log_info "Container status:"
    docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

main "$@"




