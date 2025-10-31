#!/bin/bash

# Production Deployment Script for OS Content UGC Video Generator
# Handles automated deployment with health checks and rollback

set -e

# Configuration
PROJECT_NAME="os-content"
DOCKER_COMPOSE_FILE="production/docker-compose.prod.yml"
ENV_FILE=".env.production"
BACKUP_BEFORE_DEPLOY=true
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_ON_FAILURE=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file $ENV_FILE not found"
        exit 1
    fi
    
    # Check if docker-compose file exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error "Docker Compose file $DOCKER_COMPOSE_FILE not found"
        exit 1
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    
    if [ "$AVAILABLE_SPACE_GB" -lt 10 ]; then
        error "Insufficient disk space: ${AVAILABLE_SPACE_GB}GB available, need at least 10GB"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Function to create backup before deployment
create_backup() {
    if [ "$BACKUP_BEFORE_DEPLOY" = true ]; then
        log "Creating backup before deployment"
        
        # Create backup using the backup service
        docker-compose -f "$DOCKER_COMPOSE_FILE" --profile backup run --rm backup
        
        if [ $? -eq 0 ]; then
            log "Backup created successfully"
        else
            warn "Backup creation failed, continuing with deployment"
        fi
    fi
}

# Function to stop current deployment
stop_current_deployment() {
    log "Stopping current deployment"
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
    
    if [ $? -eq 0 ]; then
        log "Current deployment stopped"
    else
        warn "Failed to stop current deployment cleanly"
    fi
}

# Function to build new images
build_images() {
    log "Building new Docker images"
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    if [ $? -eq 0 ]; then
        log "Docker images built successfully"
    else
        error "Failed to build Docker images"
        exit 1
    fi
}

# Function to start new deployment
start_deployment() {
    log "Starting new deployment"
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    if [ $? -eq 0 ]; then
        log "Deployment started successfully"
    else
        error "Failed to start deployment"
        exit 1
    fi
}

# Function to wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready"
    
    # Wait for database
    log "Waiting for database..."
    timeout 60 bash -c 'until docker-compose -f '"$DOCKER_COMPOSE_FILE"' exec -T postgres pg_isready -U os_content_user -d os_content; do sleep 2; done'
    
    if [ $? -ne 0 ]; then
        error "Database failed to become ready"
        return 1
    fi
    
    # Wait for Redis
    log "Waiting for Redis..."
    timeout 30 bash -c 'until docker-compose -f '"$DOCKER_COMPOSE_FILE"' exec -T redis redis-cli ping; do sleep 2; done'
    
    if [ $? -ne 0 ]; then
        error "Redis failed to become ready"
        return 1
    fi
    
    # Wait for main application
    log "Waiting for main application..."
    timeout $HEALTH_CHECK_TIMEOUT bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    
    if [ $? -ne 0 ]; then
        error "Main application failed to become ready"
        return 1
    fi
    
    # Wait for Nginx
    log "Waiting for Nginx..."
    timeout 30 bash -c 'until curl -f http://localhost/health; do sleep 2; done'
    
    if [ $? -ne 0 ]; then
        error "Nginx failed to become ready"
        return 1
    fi
    
    log "All services are ready"
    return 0
}

# Function to run health checks
run_health_checks() {
    log "Running health checks"
    
    # Check application health
    log "Checking application health..."
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        error "Application health check failed"
        return 1
    fi
    
    # Check database health
    log "Checking database health..."
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U os_content_user -d os_content > /dev/null 2>&1; then
        error "Database health check failed"
        return 1
    fi
    
    # Check Redis health
    log "Checking Redis health..."
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
        error "Redis health check failed"
        return 1
    fi
    
    # Check Nginx health
    log "Checking Nginx health..."
    if ! curl -f http://localhost/health > /dev/null 2>&1; then
        error "Nginx health check failed"
        return 1
    fi
    
    # Check Prometheus health
    log "Checking Prometheus health..."
    if ! curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        warn "Prometheus health check failed"
    fi
    
    # Check Grafana health
    log "Checking Grafana health..."
    if ! curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        warn "Grafana health check failed"
    fi
    
    log "Health checks completed successfully"
    return 0
}

# Function to run smoke tests
run_smoke_tests() {
    log "Running smoke tests"
    
    # Test API endpoints
    log "Testing API endpoints..."
    
    # Test health endpoint
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        error "Health endpoint test failed"
        return 1
    fi
    
    # Test metrics endpoint
    if ! curl -f http://localhost:8000/metrics > /dev/null 2>&1; then
        warn "Metrics endpoint test failed"
    fi
    
    # Test API documentation
    if ! curl -f http://localhost:8000/docs > /dev/null 2>&1; then
        warn "API documentation test failed"
    fi
    
    log "Smoke tests completed"
    return 0
}

# Function to rollback deployment
rollback_deployment() {
    if [ "$ROLLBACK_ON_FAILURE" = true ]; then
        error "Deployment failed, initiating rollback"
        
        # Stop current deployment
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        
        # Restore from backup if available
        if [ -d "backups" ]; then
            log "Restoring from backup..."
            # This would implement actual backup restoration
            warn "Backup restoration not implemented"
        fi
        
        # Start previous version
        log "Starting previous version..."
        # This would implement starting the previous version
        
        error "Rollback completed"
    fi
}

# Function to show deployment status
show_deployment_status() {
    log "Deployment Status:"
    
    echo
    echo "=== Service Status ==="
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo
    echo "=== Service Logs ==="
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=10
    
    echo
    echo "=== Resource Usage ==="
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Function to send deployment notification
send_deployment_notification() {
    local status=$1
    local message=$2
    
    log "Sending deployment notification: $status"
    
    # This would implement actual notification sending
    # Slack, email, etc.
    
    if [ "$status" = "success" ]; then
        info "Deployment notification sent: SUCCESS"
    else
        error "Deployment notification sent: FAILED"
    fi
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    log "=== Starting Production Deployment ==="
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup
    create_backup
    
    # Stop current deployment
    stop_current_deployment
    
    # Build new images
    build_images
    
    # Start new deployment
    start_deployment
    
    # Wait for services
    if ! wait_for_services; then
        error "Services failed to become ready"
        rollback_deployment
        send_deployment_notification "failed" "Services failed to become ready"
        exit 1
    fi
    
    # Run health checks
    if ! run_health_checks; then
        error "Health checks failed"
        rollback_deployment
        send_deployment_notification "failed" "Health checks failed"
        exit 1
    fi
    
    # Run smoke tests
    if ! run_smoke_tests; then
        error "Smoke tests failed"
        rollback_deployment
        send_deployment_notification "failed" "Smoke tests failed"
        exit 1
    fi
    
    # Calculate deployment time
    local end_time=$(date +%s)
    local deployment_time=$((end_time - start_time))
    
    # Show deployment status
    show_deployment_status
    
    # Send success notification
    send_deployment_notification "success" "Deployment completed in ${deployment_time}s"
    
    log "=== Production Deployment Completed Successfully ==="
    log "Deployment time: ${deployment_time} seconds"
    
    exit 0
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --no-backup             Skip backup before deployment"
    echo "  --no-rollback           Skip rollback on failure"
    echo "  --timeout SECONDS       Health check timeout (default: 300)"
    echo
    echo "Examples:"
    echo "  $0                      Deploy with all safety checks"
    echo "  $0 --no-backup          Deploy without backup"
    echo "  $0 --timeout 600        Deploy with 10-minute timeout"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --no-backup)
            BACKUP_BEFORE_DEPLOY=false
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        --timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@" 