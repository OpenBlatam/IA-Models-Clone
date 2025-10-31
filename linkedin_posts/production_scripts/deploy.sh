#!/bin/bash

# LinkedIn Posts Production Deployment Script
# ==========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="linkedin-posts"
DOCKER_COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.production"
BACKUP_DIR="backups"
LOG_DIR="logs"

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

check_requirements() {
    log_info "Checking deployment requirements..."
    
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
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file $ENV_FILE not found"
        exit 1
    fi
    
    log_success "Requirements check passed"
}

create_backup() {
    log_info "Creating backup..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Backup timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_NAME="${PROJECT_NAME}_backup_${TIMESTAMP}"
    
    # Create backup
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_dump -U linkedin_user linkedin_posts > "$BACKUP_DIR/${BACKUP_NAME}.sql"
    
    # Compress backup
    gzip "$BACKUP_DIR/${BACKUP_NAME}.sql"
    
    log_success "Backup created: $BACKUP_DIR/${BACKUP_NAME}.sql.gz"
}

stop_services() {
    log_info "Stopping existing services..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
    
    log_success "Services stopped"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build with no cache for production
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    log_success "Images built successfully"
}

start_services() {
    log_info "Starting services..."
    
    # Start services in background
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_success "Services started"
}

wait_for_health() {
    log_info "Waiting for services to be healthy..."
    
    # Wait for API to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - Waiting for API to be ready..."
        sleep 10
        ((attempt++))
    done
    
    log_error "API failed to become healthy after $max_attempts attempts"
    return 1
}

run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for database to be ready
    sleep 10
    
    # Run migrations
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T linkedin-posts-api python -m alembic upgrade head
    
    log_success "Migrations completed"
}

check_services() {
    log_info "Checking service status..."
    
    # Check all services
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    # Check API health
    if curl -f http://localhost:8000/health; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Check database
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U linkedin_user; then
        log_success "Database health check passed"
    else
        log_error "Database health check failed"
        return 1
    fi
    
    # Check Redis
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
        return 1
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    # Keep only last 10 backups
    ls -t "$BACKUP_DIR"/*.sql.gz 2>/dev/null | tail -n +11 | xargs -r rm
    
    log_success "Old backups cleaned up"
}

show_deployment_info() {
    log_success "Deployment completed successfully!"
    echo
    echo "=== Deployment Information ==="
    echo "API URL: http://localhost:8000"
    echo "API Docs: http://localhost:8000/docs"
    echo "Health Check: http://localhost:8000/health"
    echo "Grafana: http://localhost:3000 (admin/admin)"
    echo "Kibana: http://localhost:5601"
    echo "Prometheus: http://localhost:9090"
    echo
    echo "=== Service Status ==="
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    echo
    echo "=== Recent Logs ==="
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=20 linkedin-posts-api
}

rollback() {
    log_warning "Rolling back deployment..."
    
    # Stop current services
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    # Restore from backup if available
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/*.sql.gz 2>/dev/null | head -n 1)
    if [ -n "$LATEST_BACKUP" ]; then
        log_info "Restoring from backup: $LATEST_BACKUP"
        gunzip -c "$LATEST_BACKUP" | docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U linkedin_user linkedin_posts
    fi
    
    # Start previous version
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_success "Rollback completed"
}

# Main deployment function
deploy() {
    log_info "Starting LinkedIn Posts production deployment..."
    
    # Check requirements
    check_requirements
    
    # Create backup
    create_backup
    
    # Stop existing services
    stop_services
    
    # Build new images
    build_images
    
    # Start services
    start_services
    
    # Wait for health
    if ! wait_for_health; then
        log_error "Deployment failed - services not healthy"
        rollback
        exit 1
    fi
    
    # Run migrations
    run_migrations
    
    # Check services
    if ! check_services; then
        log_error "Deployment failed - service checks failed"
        rollback
        exit 1
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Show deployment info
    show_deployment_info
}

# Command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "status")
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        ;;
    "logs")
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f "${2:-linkedin-posts-api}"
        ;;
    "backup")
        create_backup
        ;;
    "health")
        check_services
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|logs|backup|health}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy the application (default)"
        echo "  rollback - Rollback to previous version"
        echo "  status   - Show service status"
        echo "  logs     - Show logs (optional: service name)"
        echo "  backup   - Create backup"
        echo "  health   - Check service health"
        exit 1
        ;;
esac 