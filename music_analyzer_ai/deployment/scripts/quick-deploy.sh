#!/bin/bash
# Quick Deploy Script
# Simplified deployment for common scenarios

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"

# Initialize
init_common

# Configuration
readonly DEPLOYMENT_TYPE="${1:-docker}"
readonly ENVIRONMENT="${ENVIRONMENT:-production}"

# Quick Docker deployment
quick_docker_deploy() {
    log_info "Quick Docker deployment..."
    
    validate_command docker docker.io
    validate_command docker-compose docker-compose
    
    if ! docker_is_running; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Use docker-compose
    docker_compose_up "docker-compose.yml"
    
    # Health check
    health_check "http://localhost:8000/health" 10 5
    
    log_success "Quick Docker deployment completed"
}

# Quick Kubernetes deployment
quick_k8s_deploy() {
    log_info "Quick Kubernetes deployment..."
    
    source "${SCRIPT_DIR}/lib/kubernetes.sh"
    
    if ! k8s_check_kubectl; then
        log_error "kubectl not available"
        exit 1
    fi
    
    local namespace="${ENVIRONMENT}"
    k8s_check_namespace "${namespace}"
    
    # Apply manifests
    if [ -d "kubernetes" ]; then
        k8s_apply_manifest "kubernetes/deployment.yaml"
        k8s_wait_for_deployment "music-analyzer-ai-backend" "${namespace}"
    else
        log_error "Kubernetes manifests not found"
        exit 1
    fi
    
    log_success "Quick Kubernetes deployment completed"
}

# Main function
main() {
    case "${DEPLOYMENT_TYPE}" in
        docker)
            quick_docker_deploy
            ;;
        kubernetes|k8s)
            quick_k8s_deploy
            ;;
        *)
            echo "Usage: $0 {docker|kubernetes}"
            echo ""
            echo "Quick deployment options:"
            echo "  docker      - Deploy using Docker Compose"
            echo "  kubernetes  - Deploy to Kubernetes"
            exit 1
            ;;
    esac
}

main "$@"




