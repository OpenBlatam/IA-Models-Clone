#!/bin/bash
# Deployment script for Suno Clone AI on EC2
# Follows DevOps best practices with error handling, logging, and idempotency

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Configuration
readonly CONTAINER_NAME="suno-clone-ai"
readonly IMAGE_NAME="suno-clone-ai"
readonly APP_PORT=8020
readonly HEALTH_CHECK_URL="http://localhost:${APP_PORT}/health"
readonly MAX_RETRIES=5
readonly RETRY_DELAY=10

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
trap 'log_error "Deployment failed at line $LINENO. Exit code: $?"' ERR

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_info "Docker is installed and running"
}

# Load Docker image
load_image() {
    local image_file="${IMAGE_NAME}.tar.gz"
    
    if [ ! -f "${image_file}" ]; then
        log_error "Docker image file ${image_file} not found"
        exit 1
    fi
    
    log_info "Loading Docker image from ${image_file}..."
    docker load < "${image_file}" || {
        log_error "Failed to load Docker image"
        exit 1
    }
    
    log_info "Docker image loaded successfully"
}

# Stop and remove existing container
stop_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_info "Stopping existing container ${CONTAINER_NAME}..."
        docker stop "${CONTAINER_NAME}" || true
        
        log_info "Removing existing container ${CONTAINER_NAME}..."
        docker rm "${CONTAINER_NAME}" || true
        
        log_info "Container ${CONTAINER_NAME} stopped and removed"
    else
        log_info "No existing container found"
    fi
}

# Clean up old images (keep last 2)
cleanup_images() {
    log_info "Cleaning up old Docker images..."
    docker images "${IMAGE_NAME}" --format "{{.ID}}" | tail -n +3 | while read -r image_id; do
        if [ -n "${image_id}" ]; then
            log_info "Removing old image: ${image_id}"
            docker rmi "${image_id}" || true
        fi
    done
}

# Check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        log_warn ".env file not found. Using default environment variables"
        if [ -f ".env.example" ]; then
            log_info "Copying .env.example to .env"
            cp .env.example .env
        fi
    else
        log_info ".env file found"
    fi
}

# Run container
run_container() {
    log_info "Starting container ${CONTAINER_NAME}..."
    
    local env_file_arg=""
    if [ -f ".env" ]; then
        env_file_arg="--env-file .env"
    fi
    
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        -p "${APP_PORT}:${APP_PORT}" \
        ${env_file_arg} \
        --health-cmd="curl -f ${HEALTH_CHECK_URL} || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        --health-start-period=40s \
        "${IMAGE_NAME}:latest" || {
        log_error "Failed to start container"
        docker logs "${CONTAINER_NAME}" --tail 50 || true
        exit 1
    }
    
    log_info "Container ${CONTAINER_NAME} started"
}

# Wait for health check
wait_for_health() {
    log_info "Waiting for application to be healthy..."
    
    local retries=0
    while [ ${retries} -lt ${MAX_RETRIES} ]; do
        if curl -f "${HEALTH_CHECK_URL}" &> /dev/null; then
            log_info "Health check passed"
            return 0
        fi
        
        retries=$((retries + 1))
        if [ ${retries} -lt ${MAX_RETRIES} ]; then
            log_warn "Health check failed, retrying in ${RETRY_DELAY} seconds... (${retries}/${MAX_RETRIES})"
            sleep ${RETRY_DELAY}
        fi
    done
    
    log_error "Health check failed after ${MAX_RETRIES} attempts"
    docker logs "${CONTAINER_NAME}" --tail 50
    return 1
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container ${CONTAINER_NAME} is not running"
        docker logs "${CONTAINER_NAME}" --tail 50
        exit 1
    fi
    
    # Check health endpoint
    if ! curl -f "${HEALTH_CHECK_URL}" &> /dev/null; then
        log_error "Health check failed"
        docker logs "${CONTAINER_NAME}" --tail 50
        exit 1
    fi
    
    log_info "Deployment verified successfully"
}

# Main deployment function
main() {
    log_info "Starting Suno Clone AI deployment..."
    
    check_docker
    check_env_file
    load_image
    stop_container
    cleanup_images
    run_container
    wait_for_health
    verify_deployment
    
    log_info "✅ Deployment completed successfully!"
    
    # Display container status
    docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Run main function
main "$@"




