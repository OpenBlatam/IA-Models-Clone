#!/bin/bash
# Enterprise Deployment Script for Music Analyzer AI
# Supports multiple deployment strategies: blue-green, canary, rolling
# Refactored with modular design and reusable libraries

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"

# Initialize
init_common

# Configuration
readonly PROJECT_ROOT=$(get_project_root)
readonly DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-rolling}"
readonly ENVIRONMENT="${ENVIRONMENT:-production}"
readonly BACKEND_IMAGE="${BACKEND_IMAGE:-music-analyzer-ai-backend:latest}"
readonly FRONTEND_IMAGE="${FRONTEND_IMAGE:-music-analyzer-ai-frontend:latest}"
readonly HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"
readonly MAX_RETRIES=10
readonly RETRY_DELAY=5

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Validate required commands
    validate_command docker docker.io
    validate_command docker-compose docker-compose
    validate_command curl curl
    validate_command jq jq || log_warn "jq not found, some features may be limited"
    
    # Validate Docker daemon
    if ! docker_is_running; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Validate required environment variables
    validate_required_vars ENVIRONMENT
    
    log_success "All prerequisites met"
}

# Health check function (using common library)
perform_health_check() {
    local url="${1:-${HEALTH_CHECK_URL}}"
    health_check "${url}" "${MAX_RETRIES}" "${RETRY_DELAY}"
}

# Rolling deployment
deploy_rolling() {
    log_info "Starting rolling deployment..."
    
    local compose_file="${PROJECT_ROOT}/deployment/docker-compose.yml"
    validate_file_exists "${compose_file}"
    
    # Stop old containers gracefully
    docker_compose_down "${compose_file}" || true
    
    # Start new containers
    docker_compose_up "${compose_file}" "backend frontend"
    
    # Wait for health check
    perform_health_check
    
    log_success "Rolling deployment completed"
}

# Blue-Green deployment
deploy_blue_green() {
    log_info "Starting blue-green deployment..."
    
    local blue_backend="music-analyzer-ai-backend-blue"
    local green_backend="music-analyzer-ai-backend-green"
    local current_backend
    
    # Determine current environment
    if docker ps --format '{{.Names}}' | grep -q "${blue_backend}"; then
        current_backend="${blue_backend}"
        new_backend="${green_backend}"
        current_color="blue"
        new_color="green"
    else
        current_backend="${green_backend}"
        new_backend="${blue_backend}"
        current_color="green"
        new_color="blue"
    fi
    
    log_info "Current environment: ${current_color}, deploying to: ${new_color}"
    
    # Deploy new environment
    BACKEND_CONTAINER_NAME="${new_backend}" \
    docker-compose -f docker-compose.yml -f docker-compose.blue-green.yml \
        up -d --build backend
    
    # Health check on new environment
    local new_health_url="${HEALTH_CHECK_URL/8000/8001}"
    health_check "${new_health_url}"
    
    # Switch traffic (update load balancer/service)
    log_info "Switching traffic to ${new_color} environment..."
    # This would typically update a load balancer or service configuration
    
    # Stop old environment
    log_info "Stopping ${current_color} environment..."
    docker stop "${current_backend}" || true
    docker rm "${current_backend}" || true
    
    log_success "Blue-green deployment completed"
}

# Canary deployment
deploy_canary() {
    log_info "Starting canary deployment..."
    
    local canary_traffic_percent="${CANARY_TRAFFIC_PERCENT:-10}"
    
    log_info "Deploying canary with ${canary_traffic_percent}% traffic"
    
    # Deploy canary version
    CANARY=true \
    docker-compose -f docker-compose.yml -f docker-compose.canary.yml \
        up -d --scale backend-canary=1
    
    # Health check canary
    health_check "http://localhost:8002/health"
    
    # Monitor canary metrics
    log_info "Monitoring canary metrics for 5 minutes..."
    sleep 300
    
    # If canary is healthy, gradually increase traffic
    if health_check "http://localhost:8002/health"; then
        log_info "Canary is healthy, proceeding with full deployment..."
        deploy_rolling
    else
        log_error "Canary deployment failed, rolling back..."
        docker-compose -f docker-compose.canary.yml down
        exit 1
    fi
    
    log_success "Canary deployment completed"
}

# Main deployment function
main() {
    log_info "Starting Music Analyzer AI deployment"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Strategy: ${DEPLOYMENT_STRATEGY}"
    
    validate_prerequisites
    
    cd "${PROJECT_ROOT}/deployment"
    
    case "${DEPLOYMENT_STRATEGY}" in
        rolling)
            deploy_rolling
            ;;
        blue-green)
            deploy_blue_green
            ;;
        canary)
            deploy_canary
            ;;
        *)
            log_error "Unknown deployment strategy: ${DEPLOYMENT_STRATEGY}"
            log_info "Available strategies: rolling, blue-green, canary"
            exit 1
            ;;
    esac
    
    # Final verification
    log_info "Performing final verification..."
    health_check
    
    # Display deployment status
    log_info "Deployment status:"
    docker-compose ps
    
    log_success "Deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --backend-image)
            BACKEND_IMAGE="$2"
            shift 2
            ;;
        --frontend-image)
            FRONTEND_IMAGE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --strategy STRATEGY     Deployment strategy (rolling|blue-green|canary)"
            echo "  --environment ENV      Environment (development|staging|production)"
            echo "  --backend-image IMAGE  Backend Docker image"
            echo "  --frontend-image IMAGE Frontend Docker image"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"

