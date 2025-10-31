#!/bin/bash

# Production Deployment Script for Blaze AI
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="blaze-ai"
DEPLOYMENT_DIR="deployment/kubernetes"
DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.prod.yml"

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

check_kubernetes_context() {
    log_info "Checking Kubernetes context..."
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    CURRENT_CONTEXT=$(kubectl config current-context)
    log_info "Current Kubernetes context: $CURRENT_CONTEXT"
    
    read -p "Do you want to continue with this context? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace and RBAC
    kubectl apply -f $DEPLOYMENT_DIR/namespace.yaml
    
    # Create storage
    kubectl apply -f $DEPLOYMENT_DIR/storage.yaml
    
    # Create secrets
    kubectl apply -f $DEPLOYMENT_DIR/secrets.yaml
    
    # Create monitoring
    kubectl apply -f $DEPLOYMENT_DIR/monitoring.yaml
    
    # Create deployment
    kubectl apply -f $DEPLOYMENT_DIR/deployment.yaml
    
    # Create ingress
    kubectl apply -f $DEPLOYMENT_DIR/ingress.yaml
    
    log_info "Kubernetes deployment completed"
}

deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_warn ".env file not found, creating from template..."
        cat > .env << EOF
# Database Configuration
DB_PASSWORD=your_secure_password_here
REDIS_PASSWORD=your_redis_password_here

# API Configuration
BLAZE_AI_API_KEY=your_api_key_here
BLAZE_AI_JWT_SECRET=your_jwt_secret_here

# SSL Configuration
SSL_CERT_PATH=./deployment/nginx/ssl/cert.pem
SSL_KEY_PATH=./deployment/nginx/ssl/key.pem
EOF
        log_warn "Please update .env file with your actual values"
        exit 1
    fi
    
    # Build and start services
    docker-compose -f $DOCKER_COMPOSE_FILE up -d --build
    
    log_info "Docker Compose deployment completed"
}

check_deployment_status() {
    log_info "Checking deployment status..."
    
    if [ "$1" = "kubernetes" ]; then
        kubectl get pods -n $NAMESPACE
        kubectl get services -n $NAMESPACE
        kubectl get ingress -n $NAMESPACE
    else
        docker-compose -f $DOCKER_COMPOSE_FILE ps
    fi
}

show_access_info() {
    log_info "Access Information:"
    
    if [ "$1" = "kubernetes" ]; then
        echo "API: https://api.blazeai.com"
        echo "Gradio: https://gradio.blazeai.com"
        echo "Metrics: https://metrics.blazeai.com"
        echo "Prometheus: http://localhost:9090"
        echo "Grafana: http://localhost:3000"
    else
        echo "API: http://localhost:8000"
        echo "Gradio: http://localhost:8001"
        echo "Metrics: http://localhost:8002"
        echo "Nginx: http://localhost (redirects to HTTPS)"
    fi
}

# Main deployment logic
main() {
    log_info "Starting Blaze AI production deployment..."
    
    check_prerequisites
    
    echo "Choose deployment method:"
    echo "1) Kubernetes"
    echo "2) Docker Compose"
    read -p "Enter your choice (1 or 2): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            check_kubernetes_context
            deploy_kubernetes
            check_deployment_status "kubernetes"
            show_access_info "kubernetes"
            ;;
        2)
            deploy_docker
            check_deployment_status "docker"
            show_access_info "docker"
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
    
    log_info "Deployment completed successfully!"
}

# Run main function
main "$@"
