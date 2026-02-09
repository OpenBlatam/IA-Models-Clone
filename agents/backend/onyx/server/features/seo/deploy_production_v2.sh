#!/bin/bash

# Ultra-Optimized SEO Service v2.0 - Production Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="seo-service-v2"
DOCKER_COMPOSE_FILE="docker-compose.production_v2.yml"
ENV_FILE=".env.production"

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
    log_info "Checking system requirements..."
    
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
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 4 ]; then
        log_warning "Recommended minimum 4GB RAM, found ${MEMORY_GB}GB"
    fi
    
    # Check available disk space
    DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$DISK_GB" -lt 10 ]; then
        log_warning "Recommended minimum 10GB free space, found ${DISK_GB}GB"
    fi
    
    log_success "System requirements check completed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create environment file if not exists
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment file..."
        cat > "$ENV_FILE" << EOF
# Ultra-Optimized SEO Service v2.0 - Production Environment

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 16)
GRAFANA_PASSWORD=$(openssl rand -hex 16)

# Service Configuration
ENVIRONMENT=production
DEBUG=false
HOST=0.0.0.0
PORT=8000
WORKERS=4
MAX_CONCURRENT_REQUESTS=1000
MAX_REQUESTS_PER_WORKER=10000

# Database
POSTGRES_DB=seo_service
POSTGRES_USER=seo_user

# Redis
REDIS_URL=redis://redis:6379/0

# CORS
CORS_ORIGINS=["*"]

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/seo-service.log

# Cache
CACHE_TTL=3600
CACHE_SIZE=1000

# Rate Limiting
RATE_LIMIT=200
RATE_LIMIT_WINDOW=1.0

# Compression
ENABLE_COMPRESSION=true
COMPRESSION_LEVEL=3
COMPRESSION_THRESHOLD=1024

# HTTP Client
HTTP_TIMEOUT=15.0
HTTP_MAX_CONNECTIONS=200
HTTP_MAX_KEEPALIVE=20
HTTP_KEEPALIVE_EXPIRY=30.0
HTTP_ENABLE_HTTP2=true
HTTP_RETRY_ATTEMPTS=3
HTTP_MAX_REDIRECTS=5

# Parser
PARSER_TIMEOUT=10.0
PARSER_MAX_SIZE=52428800
PARSER_ENABLE_COMPRESSION=true
PARSER_COMPRESSION_LEVEL=3

# SEO Analysis
SEO_MIN_TITLE_LENGTH=30
SEO_MAX_TITLE_LENGTH=60
SEO_MIN_DESCRIPTION_LENGTH=120
SEO_MAX_DESCRIPTION_LENGTH=160
SEO_MIN_CONTENT_LENGTH=300
SEO_MAX_CONCURRENT_ANALYSES=10
SEO_ANALYSIS_TIMEOUT=30.0
EOF
        log_success "Environment file created"
    else
        log_info "Environment file already exists"
    fi
    
    # Create necessary directories
    mkdir -p logs cache data ssl grafana/dashboards grafana/datasources
    
    log_success "Environment setup completed"
}

setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        log_info "Generating self-signed SSL certificates..."
        mkdir -p ssl
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        chmod 600 ssl/key.pem
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'seo-service'
    static_configs:
      - targets: ['seo-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
    
    # Grafana datasource
    mkdir -p grafana/datasources
    cat > grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    # Filebeat configuration
    cat > filebeat.yml << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/seo/*.log
  json.keys_under_root: true
  json.add_error_key: true
  json.message_key: message

processors:
  - add_host_metadata: ~
  - add_cloud_metadata: ~

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  indices:
    - index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"

setup.kibana:
  host: "kibana:5601"
EOF
    
    log_success "Monitoring configuration completed"
}

build_images() {
    log_info "Building Docker images..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    log_success "Docker images built successfully"
}

deploy_services() {
    log_info "Deploying services..."
    
    # Stop existing services
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_success "Services deployed successfully"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL..."
    timeout=60
    while ! docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U seo_user -d seo_service > /dev/null 2>&1; do
        sleep 1
        timeout=$((timeout - 1))
        if [ $timeout -le 0 ]; then
            log_error "PostgreSQL failed to start within 60 seconds"
            exit 1
        fi
    done
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    timeout=30
    while ! docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; do
        sleep 1
        timeout=$((timeout - 1))
        if [ $timeout -le 0 ]; then
            log_error "Redis failed to start within 30 seconds"
            exit 1
        fi
    done
    
    # Wait for SEO Service
    log_info "Waiting for SEO Service..."
    timeout=60
    while ! curl -f http://localhost:8000/health > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "SEO Service failed to start within 60 seconds"
            exit 1
        fi
    done
    
    log_success "All services are ready"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Check SEO Service
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "SEO Service is healthy"
    else
        log_error "SEO Service health check failed"
        exit 1
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
    
    # Check Kibana
    if curl -f http://localhost:5601/api/status > /dev/null 2>&1; then
        log_success "Kibana is healthy"
    else
        log_warning "Kibana health check failed"
    fi
    
    log_success "Health checks completed"
}

show_status() {
    log_info "Service Status:"
    echo ""
    echo "SEO Service:     http://localhost:8000"
    echo "API Docs:        http://localhost:8000/docs"
    echo "Health Check:    http://localhost:8000/health"
    echo "Metrics:         http://localhost:8000/metrics"
    echo ""
    echo "Grafana:         http://localhost:3000 (admin / ${GRAFANA_PASSWORD:-password})"
    echo "Prometheus:      http://localhost:9090"
    echo "Kibana:          http://localhost:5601"
    echo ""
    echo "Nginx:           http://localhost (HTTP)"
    echo "Nginx:           https://localhost (HTTPS - self-signed)"
    echo ""
    
    # Show running containers
    log_info "Running containers:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f *.tmp
    log_success "Cleanup completed"
}

main() {
    log_info "Starting Ultra-Optimized SEO Service v2.0 deployment..."
    
    check_requirements
    setup_environment
    setup_ssl
    setup_monitoring
    build_images
    deploy_services
    wait_for_services
    run_health_checks
    show_status
    cleanup
    
    log_success "Deployment completed successfully!"
    echo ""
    log_info "Next steps:"
    echo "1. Access the API at http://localhost:8000"
    echo "2. View metrics in Grafana at http://localhost:3000"
    echo "3. Check logs in Kibana at http://localhost:5601"
    echo "4. Monitor with Prometheus at http://localhost:9090"
    echo ""
    log_info "To stop services: docker-compose -f $DOCKER_COMPOSE_FILE down"
    log_info "To view logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
}

# Run main function
main "$@" 