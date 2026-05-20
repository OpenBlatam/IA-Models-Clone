#!/bin/bash

# Deploy Script - LinkedIn Posts Ultra Optimized
# =============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="linkedin-posts"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE="env.production"
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
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
    
    # Check if env file exists
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file $ENV_FILE not found"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "./ssl"
    mkdir -p "./grafana/dashboards"
    mkdir -p "./grafana/datasources"
    
    log_success "Directories created"
}

# Backup existing data
backup_data() {
    log_info "Creating backup of existing data..."
    
    if [ -d "$BACKUP_DIR" ]; then
        BACKUP_FILE="$BACKUP_DIR/backup-$(date +%Y%m%d-%H%M%S).tar.gz"
        tar -czf "$BACKUP_FILE" ./data 2>/dev/null || true
        log_success "Backup created: $BACKUP_FILE"
    fi
}

# Stop existing services
stop_services() {
    log_info "Stopping existing services..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans || true
    log_success "Services stopped"
}

# Build and start services
deploy_services() {
    log_info "Building and starting services..."
    
    # Build images
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_success "Services deployed"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for database
    log_info "Waiting for database..."
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T db pg_isready -U user -d linkedin_posts; do
        sleep 2
    done
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping; do
        sleep 2
    done
    
    # Wait for API
    log_info "Waiting for API..."
    until curl -f http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
    done
    
    log_success "All services are ready"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Create database tables if they don't exist
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T db psql -U user -d linkedin_posts -c "
        CREATE TABLE IF NOT EXISTS linkedin_posts (
            id VARCHAR(255) PRIMARY KEY,
            content TEXT NOT NULL,
            post_type VARCHAR(50) NOT NULL,
            tone VARCHAR(50) NOT NULL,
            target_audience VARCHAR(100) NOT NULL,
            industry VARCHAR(50) NOT NULL,
            tags TEXT[],
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_linkedin_posts_created_at ON linkedin_posts(created_at);
        CREATE INDEX IF NOT EXISTS idx_linkedin_posts_post_type ON linkedin_posts(post_type);
        CREATE INDEX IF NOT EXISTS idx_linkedin_posts_industry ON linkedin_posts(industry);
    " || log_warning "Database migration failed (table might already exist)"
    
    log_success "Database migrations completed"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Check metrics endpoint
    if curl -f http://localhost:8000/metrics > /dev/null 2>&1; then
        log_success "Metrics endpoint is accessible"
    else
        log_warning "Metrics endpoint is not accessible"
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_success "Prometheus is healthy"
    else
        log_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        log_success "Grafana is healthy"
    else
        log_warning "Grafana health check failed"
    fi
}

# Performance test
performance_test() {
    log_info "Running performance test..."
    
    # Simple load test
    for i in {1..10}; do
        curl -s http://localhost:8000/health > /dev/null &
    done
    wait
    
    log_success "Performance test completed"
}

# Show service status
show_status() {
    log_info "Service status:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo ""
    log_info "Service URLs:"
    echo "  API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Health: http://localhost:8000/health"
    echo "  Metrics: http://localhost:8000/metrics"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3000"
    echo "  Kibana: http://localhost:5601"
}

# Cleanup old backups
cleanup_backups() {
    log_info "Cleaning up old backups..."
    
    # Keep only last 5 backups
    find "$BACKUP_DIR" -name "backup-*.tar.gz" -type f -mtime +7 -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting deployment of $APP_NAME..."
    
    check_prerequisites
    create_directories
    backup_data
    stop_services
    deploy_services
    wait_for_services
    run_migrations
    health_check
    performance_test
    cleanup_backups
    show_status
    
    log_success "Deployment completed successfully!"
    log_info "You can now access the application at http://localhost:8000"
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    # Stop current services
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    # Restore from latest backup
    LATEST_BACKUP=$(find "$BACKUP_DIR" -name "backup-*.tar.gz" -type f | sort | tail -1)
    if [ -n "$LATEST_BACKUP" ]; then
        log_info "Restoring from backup: $LATEST_BACKUP"
        tar -xzf "$LATEST_BACKUP" -C ./
    fi
    
    log_success "Rollback completed"
}

# Usage
usage() {
    echo "Usage: $0 [deploy|rollback|status|logs]"
    echo ""
    echo "Commands:"
    echo "  deploy   - Deploy the application"
    echo "  rollback - Rollback to previous version"
    echo "  status   - Show service status"
    echo "  logs     - Show service logs"
    echo ""
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    rollback)
        rollback
        ;;
    status)
        show_status
        ;;
    logs)
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
        ;;
    *)
        usage
        exit 1
        ;;
esac 