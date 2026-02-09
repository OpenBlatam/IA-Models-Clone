#!/bin/bash

# Ultra-Optimized Production Deployment Script v10
# Maximum Performance Deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="seo-service-v10"
DOCKER_COMPOSE_FILE="docker-compose.production_v10.yml"
ENVIRONMENT="production"
BACKUP_DIR="./backups"
LOG_DIR="./logs"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check required files
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
    
    if [ ! -f "Dockerfile.production_v10" ]; then
        log_error "Dockerfile not found: Dockerfile.production_v10"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "./ssl"
    mkdir -p "./grafana/dashboards"
    mkdir -p "./grafana/datasources"
    
    log_success "Directories created"
}

backup_existing() {
    if docker ps -q -f name="$PROJECT_NAME" | grep -q .; then
        log_info "Backing up existing deployment..."
        
        # Create backup timestamp
        BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
        
        mkdir -p "$BACKUP_PATH"
        
        # Backup logs
        if [ -d "$LOG_DIR" ]; then
            cp -r "$LOG_DIR" "$BACKUP_PATH/"
        fi
        
        # Backup configuration files
        cp "$DOCKER_COMPOSE_FILE" "$BACKUP_PATH/"
        cp "Dockerfile.production_v10" "$BACKUP_PATH/"
        cp "requirements.ultra_optimized_v10.txt" "$BACKUP_PATH/"
        
        log_success "Backup created: $BACKUP_PATH"
    fi
}

stop_existing() {
    log_info "Stopping existing deployment..."
    
    if docker ps -q -f name="$PROJECT_NAME" | grep -q .; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
        log_success "Existing deployment stopped"
    else
        log_info "No existing deployment found"
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    # Build with no cache for fresh start
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    log_success "Docker images built successfully"
}

deploy_services() {
    log_info "Deploying services..."
    
    # Start services in background
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_success "Services deployed successfully"
}

wait_for_health() {
    log_info "Waiting for services to be healthy..."
    
    # Wait for main service
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "Main service is healthy"
            break
        fi
        
        log_info "Waiting for main service... (attempt $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Main service failed to become healthy"
        return 1
    fi
    
    # Wait for Redis
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if docker exec seo-redis-v10 redis-cli ping &> /dev/null; then
            log_success "Redis is healthy"
            break
        fi
        
        log_info "Waiting for Redis... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Redis failed to become healthy"
        return 1
    fi
}

run_tests() {
    log_info "Running health checks and tests..."
    
    # Test main endpoint
    if curl -f http://localhost:8000/ &> /dev/null; then
        log_success "Main endpoint is accessible"
    else
        log_error "Main endpoint is not accessible"
        return 1
    fi
    
    # Test health endpoint
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Health endpoint is accessible"
    else
        log_error "Health endpoint is not accessible"
        return 1
    fi
    
    # Test metrics endpoint
    if curl -f http://localhost:8000/metrics &> /dev/null; then
        log_success "Metrics endpoint is accessible"
    else
        log_error "Metrics endpoint is not accessible"
        return 1
    fi
    
    log_success "All health checks passed"
}

show_status() {
    log_info "Deployment status:"
    
    echo ""
    echo "Services:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo ""
    echo "Service URLs:"
    echo "  Main Service: http://localhost:8000"
    echo "  Health Check: http://localhost:8000/health"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Metrics: http://localhost:8000/metrics"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3000 (admin/admin)"
    
    echo ""
    echo "Logs:"
    echo "  Main Service: docker logs seo-service-v10"
    echo "  Redis: docker logs seo-redis-v10"
    echo "  Nginx: docker logs seo-nginx-v10"
}

cleanup_old_images() {
    log_info "Cleaning up old Docker images..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove dangling images
    docker image prune -a -f
    
    log_success "Docker cleanup completed"
}

monitor_deployment() {
    log_info "Monitoring deployment for 60 seconds..."
    
    for i in {1..6}; do
        echo "Check $i/6:"
        
        # Check service status
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        
        # Check resource usage
        echo "Resource usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
        
        echo ""
        sleep 10
    done
}

# Main deployment function
main() {
    log_info "Starting ultra-optimized production deployment v10..."
    
    # Check dependencies
    check_dependencies
    
    # Create directories
    create_directories
    
    # Backup existing deployment
    backup_existing
    
    # Stop existing deployment
    stop_existing
    
    # Build images
    build_images
    
    # Deploy services
    deploy_services
    
    # Wait for health checks
    wait_for_health
    
    # Run tests
    run_tests
    
    # Show status
    show_status
    
    # Monitor deployment
    monitor_deployment
    
    # Cleanup old images
    cleanup_old_images
    
    log_success "Ultra-optimized production deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping deployment..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        log_success "Deployment stopped"
        ;;
    "restart")
        log_info "Restarting deployment..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" restart
        log_success "Deployment restarted"
        ;;
    "logs")
        log_info "Showing logs..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        log_info "Cleaning up deployment..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
        docker system prune -f
        log_success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|cleanup}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the ultra-optimized production stack"
        echo "  stop     - Stop the deployment"
        echo "  restart  - Restart the deployment"
        echo "  logs     - Show deployment logs"
        echo "  status   - Show deployment status"
        echo "  cleanup  - Clean up deployment and Docker resources"
        exit 1
        ;;
esac 