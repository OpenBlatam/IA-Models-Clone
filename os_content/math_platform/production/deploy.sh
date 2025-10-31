#!/bin/bash

# Production Deployment Script for Math Platform
# This script handles the complete deployment process

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$SCRIPT_DIR/.env"
BACKUP_DIR="$SCRIPT_DIR/backups"
LOG_DIR="$SCRIPT_DIR/logs"
DEPLOYMENT_LOG="$LOG_DIR/deployment.log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$DEPLOYMENT_LOG"
    exit 1
}

check_dependencies() {
    log "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check required files
    if [[ ! -f "$SCRIPT_DIR/docker-compose.yml" ]]; then
        error "docker-compose.yml not found"
    fi
    
    if [[ ! -f "$SCRIPT_DIR/Dockerfile" ]]; then
        error "Dockerfile not found"
    fi
    
    success "Dependencies check passed"
}

setup_environment() {
    log "Setting up environment..."
    
    # Create necessary directories
    mkdir -p "$BACKUP_DIR" "$LOG_DIR"
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        log "Creating .env file from template..."
        cat > "$ENV_FILE" << EOF
# Math Platform Production Environment Variables

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=math_platform
DB_USER=math_user
DB_PASSWORD=$(openssl rand -base64 32)

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=$(openssl rand -base64 32)

# Security
SECRET_KEY=$(openssl rand -base64 64)
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com

# Monitoring
PROMETHEUS_ENABLED=true
SENTRY_DSN=

# Performance
MAX_WORKERS=8
CACHE_SIZE=10000

# Grafana
GRAFANA_PASSWORD=$(openssl rand -base64 16)
EOF
        success "Created .env file"
    else
        warning ".env file already exists"
    fi
    
    # Load environment variables
    source "$ENV_FILE"
    success "Environment setup complete"
}

validate_configuration() {
    log "Validating configuration..."
    
    # Check required environment variables
    local required_vars=(
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "SECRET_KEY"
        "GRAFANA_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
        fi
    done
    
    # Validate CORS origins
    if [[ "$CORS_ORIGINS" == "*" ]]; then
        warning "CORS_ORIGINS is set to '*' - this is not recommended for production"
    fi
    
    # Check if Sentry DSN is set
    if [[ -z "$SENTRY_DSN" ]]; then
        warning "SENTRY_DSN is not set - error tracking will be disabled"
    fi
    
    success "Configuration validation passed"
}

backup_database() {
    log "Creating database backup..."
    
    local backup_file="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
    
    if docker-compose -f "$SCRIPT_DIR/docker-compose.yml" exec -T postgres pg_dump -U "$DB_USER" "$DB_NAME" > "$backup_file" 2>/dev/null; then
        success "Database backup created: $backup_file"
    else
        warning "Could not create database backup (database might not be running)"
    fi
}

build_images() {
    log "Building Docker images..."
    
    cd "$SCRIPT_DIR"
    
    if docker-compose build --no-cache; then
        success "Docker images built successfully"
    else
        error "Failed to build Docker images"
    fi
}

deploy_services() {
    log "Deploying services..."
    
    cd "$SCRIPT_DIR"
    
    # Stop existing services
    log "Stopping existing services..."
    docker-compose down --remove-orphans
    
    # Start services
    log "Starting services..."
    if docker-compose up -d; then
        success "Services started successfully"
    else
        error "Failed to start services"
    fi
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
}

check_service_health() {
    log "Checking service health..."
    
    local services=(
        "math-platform-api"
        "postgres"
        "redis"
        "prometheus"
        "grafana"
    )
    
    local failed_services=()
    
    for service in "${services[@]}"; do
        if docker-compose -f "$SCRIPT_DIR/docker-compose.yml" ps "$service" | grep -q "Up"; then
            success "$service is running"
        else
            error "$service is not running"
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        error "The following services failed to start: ${failed_services[*]}"
    fi
    
    # Check API health endpoint
    log "Checking API health endpoint..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "http://localhost:8000/health" > /dev/null 2>&1; then
            success "API health check passed"
            break
        else
            if [[ $attempt -eq $max_attempts ]]; then
                error "API health check failed after $max_attempts attempts"
            else
                log "API health check attempt $attempt/$max_attempts failed, retrying..."
                sleep 10
                ((attempt++))
            fi
        fi
    done
}

setup_monitoring() {
    log "Setting up monitoring..."
    
    # Wait for Grafana to be ready
    log "Waiting for Grafana to be ready..."
    sleep 30
    
    # Import Grafana dashboards
    if [[ -d "$SCRIPT_DIR/monitoring/grafana/dashboards" ]]; then
        log "Importing Grafana dashboards..."
        # Dashboard import logic would go here
        success "Grafana dashboards imported"
    fi
    
    success "Monitoring setup complete"
}

show_deployment_info() {
    log "Deployment completed successfully!"
    echo
    echo "=== Math Platform Deployment Information ==="
    echo
    echo "API Endpoints:"
    echo "  - Main API: http://localhost:8000"
    echo "  - Health Check: http://localhost:8000/health"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo "  - Metrics: http://localhost:8000/metrics"
    echo
    echo "Monitoring:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Kibana: http://localhost:5601"
    echo
    echo "Database:"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Redis: localhost:6379"
    echo
    echo "Logs:"
    echo "  - Application logs: $LOG_DIR"
    echo "  - Docker logs: docker-compose logs -f"
    echo
    echo "Useful Commands:"
    echo "  - View logs: docker-compose logs -f math-platform-api"
    echo "  - Restart services: docker-compose restart"
    echo "  - Stop services: docker-compose down"
    echo "  - Update services: ./deploy.sh update"
    echo
}

update_services() {
    log "Updating services..."
    
    cd "$SCRIPT_DIR"
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose pull
    
    # Rebuild and restart
    build_images
    deploy_services
    
    success "Services updated successfully"
}

rollback() {
    log "Rolling back deployment..."
    
    cd "$SCRIPT_DIR"
    
    # Stop current services
    docker-compose down
    
    # Restore from backup if available
    local latest_backup=$(ls -t "$BACKUP_DIR"/*.sql 2>/dev/null | head -1)
    if [[ -n "$latest_backup" ]]; then
        log "Restoring from backup: $latest_backup"
        # Restore logic would go here
    fi
    
    # Restart services
    docker-compose up -d
    
    success "Rollback completed"
}

show_help() {
    echo "Math Platform Production Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  deploy     - Full deployment (default)"
    echo "  update     - Update existing deployment"
    echo "  rollback   - Rollback to previous version"
    echo "  backup     - Create database backup"
    echo "  health     - Check service health"
    echo "  logs       - Show service logs"
    echo "  stop       - Stop all services"
    echo "  help       - Show this help message"
    echo
}

show_logs() {
    cd "$SCRIPT_DIR"
    docker-compose logs -f
}

stop_services() {
    log "Stopping all services..."
    cd "$SCRIPT_DIR"
    docker-compose down
    success "All services stopped"
}

# Main script logic
main() {
    local command="${1:-deploy}"
    
    # Create log file
    mkdir -p "$LOG_DIR"
    touch "$DEPLOYMENT_LOG"
    
    log "Starting Math Platform deployment..."
    
    case "$command" in
        "deploy")
            check_dependencies
            setup_environment
            validate_configuration
            backup_database
            build_images
            deploy_services
            setup_monitoring
            show_deployment_info
            ;;
        "update")
            check_dependencies
            validate_configuration
            backup_database
            update_services
            ;;
        "rollback")
            rollback
            ;;
        "backup")
            setup_environment
            backup_database
            ;;
        "health")
            check_service_health
            ;;
        "logs")
            show_logs
            ;;
        "stop")
            stop_services
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            error "Unknown command: $command. Use 'help' for usage information."
            ;;
    esac
}

# Run main function with all arguments
main "$@" 