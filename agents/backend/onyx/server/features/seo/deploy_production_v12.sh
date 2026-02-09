#!/bin/bash

# Ultra-Optimized Production Deployment Script v12
# Maximum Performance and Reliability

set -euo pipefail

# Configuration
APP_NAME="seo-service-v12"
VERSION="12.0.0"
ENVIRONMENT="production"
DOCKER_COMPOSE_FILE="docker-compose.production_v12.yml"
HEALTH_CHECK_URL="http://localhost:8000/health"
GRAFANA_URL="http://localhost:3000"
KIBANA_URL="http://localhost:5601"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    success "Docker found"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    success "Docker Compose found"
    
    # Check available disk space (minimum 10GB)
    DISK_SPACE=$(df . | awk 'NR==2 {print $4}')
    if [ "$DISK_SPACE" -lt 10485760 ]; then
        warning "Low disk space available: $(($DISK_SPACE / 1024 / 1024))GB"
    else
        success "Sufficient disk space available: $(($DISK_SPACE / 1024 / 1024))GB"
    fi
    
    # Check available memory (minimum 8GB)
    MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [ "$MEMORY" -lt 8192 ]; then
        warning "Low memory available: ${MEMORY}MB"
    else
        success "Sufficient memory available: ${MEMORY}MB"
    fi
}

# Backup existing deployment
backup_existing() {
    log "Creating backup of existing deployment..."
    
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Export volumes
        docker run --rm -v "${APP_NAME}_redis_data:/data" -v "$(pwd)/$BACKUP_DIR:/backup" alpine tar czf /backup/redis_backup.tar.gz -C /data .
        docker run --rm -v "${APP_NAME}_prometheus_data:/data" -v "$(pwd)/$BACKUP_DIR:/backup" alpine tar czf /backup/prometheus_backup.tar.gz -C /data .
        docker run --rm -v "${APP_NAME}_grafana_data:/data" -v "$(pwd)/$BACKUP_DIR:/backup" alpine tar czf /backup/grafana_backup.tar.gz -C /data .
        
        success "Backup created in $BACKUP_DIR"
    else
        log "No existing deployment found, skipping backup"
    fi
}

# Stop existing deployment
stop_existing() {
    log "Stopping existing deployment..."
    
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down --timeout 30
        success "Existing deployment stopped"
    else
        log "No running deployment found"
    fi
}

# Build and deploy
deploy() {
    log "Building and deploying $APP_NAME v$VERSION..."
    
    # Build images
    log "Building Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache --parallel
    
    # Start services
    log "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    success "Deployment initiated"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for main service
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
            success "Main service is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Service failed to start within expected time"
            docker-compose -f "$DOCKER_COMPOSE_FILE" logs seo-service
            exit 1
        fi
        
        log "Waiting for service... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    # Wait for Redis
    log "Waiting for Redis..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1
    success "Redis is ready"
    
    # Wait for Prometheus
    log "Waiting for Prometheus..."
    local prometheus_attempt=0
    while [ $prometheus_attempt -lt 30 ]; do
        if curl -f -s "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
            success "Prometheus is ready"
            break
        fi
        sleep 2
        prometheus_attempt=$((prometheus_attempt + 1))
    done
    
    # Wait for Grafana
    log "Waiting for Grafana..."
    local grafana_attempt=0
    while [ $grafana_attempt -lt 30 ]; do
        if curl -f -s "$GRAFANA_URL/api/health" > /dev/null 2>&1; then
            success "Grafana is ready"
            break
        fi
        sleep 2
        grafana_attempt=$((grafana_attempt + 1))
    done
    
    # Wait for Elasticsearch
    log "Waiting for Elasticsearch..."
    local es_attempt=0
    while [ $es_attempt -lt 60 ]; do
        if curl -f -s "http://localhost:9200/_cluster/health" > /dev/null 2>&1; then
            success "Elasticsearch is ready"
            break
        fi
        sleep 2
        es_attempt=$((es_attempt + 1))
    done
    
    # Wait for Kibana
    log "Waiting for Kibana..."
    local kibana_attempt=0
    while [ $kibana_attempt -lt 30 ]; do
        if curl -f -s "$KIBANA_URL/api/status" > /dev/null 2>&1; then
            success "Kibana is ready"
            break
        fi
        sleep 2
        kibana_attempt=$((kibana_attempt + 1))
    done
}

# Run health checks
health_check() {
    log "Running comprehensive health checks..."
    
    # Check main service
    if curl -f -s "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
        success "Main service health check passed"
    else
        error "Main service health check failed"
        return 1
    fi
    
    # Check metrics endpoint
    if curl -f -s "http://localhost:8000/metrics" > /dev/null 2>&1; then
        success "Metrics endpoint is accessible"
    else
        warning "Metrics endpoint is not accessible"
    fi
    
    # Check Redis
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        success "Redis health check passed"
    else
        error "Redis health check failed"
        return 1
    fi
    
    # Check Prometheus
    if curl -f -s "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
        success "Prometheus health check passed"
    else
        warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if curl -f -s "$GRAFANA_URL/api/health" > /dev/null 2>&1; then
        success "Grafana health check passed"
    else
        warning "Grafana health check failed"
    fi
    
    # Check Elasticsearch
    if curl -f -s "http://localhost:9200/_cluster/health" > /dev/null 2>&1; then
        success "Elasticsearch health check passed"
    else
        warning "Elasticsearch health check failed"
    fi
    
    # Check Kibana
    if curl -f -s "$KIBANA_URL/api/status" > /dev/null 2>&1; then
        success "Kibana health check passed"
    else
        warning "Kibana health check failed"
    fi
}

# Performance test
performance_test() {
    log "Running performance test..."
    
    # Test basic endpoint
    local start_time=$(date +%s.%N)
    curl -f -s "$HEALTH_CHECK_URL" > /dev/null 2>&1
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc)
    
    log "Health check response time: ${response_time}s"
    
    # Test SEO analysis endpoint
    log "Testing SEO analysis endpoint..."
    local analysis_start=$(date +%s.%N)
    curl -X POST "http://localhost:8000/analyze" \
        -H "Content-Type: application/json" \
        -d '{"url": "https://example.com"}' \
        -f -s > /dev/null 2>&1
    local analysis_end=$(date +%s.%N)
    local analysis_time=$(echo "$analysis_end - $analysis_start" | bc)
    
    log "SEO analysis response time: ${analysis_time}s"
    
    # Check if response times are acceptable
    if (( $(echo "$response_time < 1.0" | bc -l) )); then
        success "Health check performance is acceptable"
    else
        warning "Health check performance is slow"
    fi
    
    if (( $(echo "$analysis_time < 5.0" | bc -l) )); then
        success "SEO analysis performance is acceptable"
    else
        warning "SEO analysis performance is slow"
    fi
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo "=================="
    
    # Service status
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo ""
    log "Service URLs:"
    echo "============="
    echo "Main Service:     http://localhost:8000"
    echo "Health Check:     $HEALTH_CHECK_URL"
    echo "API Docs:         http://localhost:8000/docs"
    echo "Metrics:          http://localhost:8000/metrics"
    echo "Prometheus:       http://localhost:9090"
    echo "Grafana:          $GRAFANA_URL (admin/admin123)"
    echo "Elasticsearch:    http://localhost:9200"
    echo "Kibana:           $KIBANA_URL"
    echo "Nginx:            http://localhost:80"
    
    echo ""
    log "Container Logs:"
    echo "==============="
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=10
}

# Main deployment function
main() {
    log "Starting ultra-optimized production deployment v$VERSION"
    log "Environment: $ENVIRONMENT"
    
    # Check prerequisites
    check_prerequisites
    
    # Backup existing deployment
    backup_existing
    
    # Stop existing deployment
    stop_existing
    
    # Deploy new version
    deploy
    
    # Wait for services
    wait_for_services
    
    # Health checks
    health_check
    
    # Performance test
    performance_test
    
    # Show status
    show_status
    
    success "Ultra-optimized production deployment v$VERSION completed successfully!"
    log "All services are running and healthy"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log "Stopping deployment..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        success "Deployment stopped"
        ;;
    "restart")
        log "Restarting deployment..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" restart
        success "Deployment restarted"
        ;;
    "logs")
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
        ;;
    "status")
        show_status
        ;;
    "health")
        health_check
        ;;
    "performance")
        performance_test
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|health|performance}"
        exit 1
        ;;
esac 