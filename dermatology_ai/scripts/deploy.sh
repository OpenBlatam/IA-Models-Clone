#!/bin/bash
# Deployment script for Dermatology AI Service
# Supports multiple deployment targets

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="dermatology-ai"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    log_info "Prerequisites OK"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t ${SERVICE_NAME}:${VERSION} .
    docker tag ${SERVICE_NAME}:${VERSION} ${SERVICE_NAME}:latest
    log_info "Image built successfully"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    python3 -m pytest tests/ -v --cov=. --cov-report=html
    log_info "Tests completed"
}

# Deploy to Docker
deploy_docker() {
    log_info "Deploying to Docker..."
    docker-compose up -d
    log_info "Deployed to Docker"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/
    log_info "Deployed to Kubernetes"
}

# Deploy to AWS Lambda
deploy_lambda() {
    log_info "Deploying to AWS Lambda..."
    
    if ! command -v serverless &> /dev/null; then
        log_error "Serverless Framework is not installed"
        exit 1
    fi
    
    serverless deploy --stage ${ENVIRONMENT}
    log_info "Deployed to AWS Lambda"
}

# Health check
health_check() {
    log_info "Performing health check..."
    sleep 5  # Wait for service to start
    
    if python3 scripts/health_check.py --url http://localhost:8006; then
        log_info "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Main deployment flow
main() {
    local target="${1:-docker}"
    
    log_info "Starting deployment to ${target}..."
    log_info "Service: ${SERVICE_NAME}"
    log_info "Version: ${VERSION}"
    log_info "Environment: ${ENVIRONMENT}"
    
    check_prerequisites
    build_image
    
    case $target in
        docker)
            deploy_docker
            health_check
            ;;
        kubernetes|k8s)
            deploy_kubernetes
            ;;
        lambda)
            deploy_lambda
            ;;
        *)
            log_error "Unknown deployment target: ${target}"
            log_info "Available targets: docker, kubernetes, lambda"
            exit 1
            ;;
    esac
    
    log_info "Deployment completed successfully!"
}

# Run main function
main "$@"















