#!/bin/bash

# Production deployment script for LinkedIn Posts API
set -e

echo "🚀 Starting LinkedIn Posts API Production Deployment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE=".env.production"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deployment.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file $ENV_FILE not found"
    fi
    
    # Check required environment variables
    source "$ENV_FILE"
    required_vars=("POSTGRES_PASSWORD" "SECRET_KEY" "OPENAI_API_KEY" "CORS_ORIGINS")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            error "Required environment variable $var is not set"
        fi
    done
    
    log "Prerequisites check passed ✅"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p logs
    mkdir -p ssl
    mkdir -p grafana/provisioning
    mkdir -p logstash/pipeline
    mkdir -p "$BACKUP_DIR"
    
    log "Directories created ✅"
}

# Backup existing data
backup_data() {
    log "Creating backup..."
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup database
    if docker-compose -f "$COMPOSE_FILE" ps db | grep -q "Up"; then
        log "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T db pg_dump -U postgres linkedin_posts > "$BACKUP_PATH/database.sql"
    fi
    
    # Backup Redis
    if docker-compose -f "$COMPOSE_FILE" ps redis | grep -q "Up"; then
        log "Backing up Redis..."
        docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli BGSAVE
        docker cp $(docker-compose -f "$COMPOSE_FILE" ps -q redis):/data/dump.rdb "$BACKUP_PATH/redis.rdb"
    fi
    
    log "Backup created at $BACKUP_PATH ✅"
}

# Build and deploy
deploy() {
    log "Starting deployment..."
    
    # Build images
    log "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    # Start services
    log "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    
    services=("db" "redis" "api" "nginx")
    for service in "${services[@]}"; do
        log "Waiting for $service to be healthy..."
        
        timeout=300  # 5 minutes
        elapsed=0
        
        while [ $elapsed -lt $timeout ]; do
            if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "healthy"; then
                log "$service is healthy ✅"
                break
            fi
            
            sleep 10
            elapsed=$((elapsed + 10))
            
            if [ $elapsed -eq $timeout ]; then
                error "$service failed to become healthy within $timeout seconds"
            fi
        done
    done
    
    log "All services are healthy ✅"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    docker-compose -f "$COMPOSE_FILE" exec api python -m alembic upgrade head
    
    log "Database migrations completed ✅"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check API health
    if ! curl -f http://localhost/health > /dev/null 2>&1; then
        error "API health check failed"
    fi
    
    # Check metrics endpoint
    if ! curl -f http://localhost/metrics > /dev/null 2>&1; then
        error "Metrics endpoint check failed"
    fi
    
    # Check Prometheus
    if ! curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        warn "Prometheus health check failed"
    fi
    
    # Check Grafana
    if ! curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        warn "Grafana health check failed"
    fi
    
    log "Deployment verification completed ✅"
}

# Cleanup old images
cleanup() {
    log "Cleaning up old Docker images..."
    
    docker image prune -f
    docker volume prune -f
    
    log "Cleanup completed ✅"
}

# Performance optimization
optimize_performance() {
    log "Applying performance optimizations..."
    
    # Set kernel parameters for better networking
    echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf
    echo 'net.ipv4.tcp_max_syn_backlog = 65535' | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
    
    # Optimize Docker daemon
    if [ -f /etc/docker/daemon.json ]; then
        log "Docker daemon already configured"
    else
        sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF
        sudo systemctl restart docker
    fi
    
    log "Performance optimizations applied ✅"
}

# Main deployment process
main() {
    log "Starting LinkedIn Posts API deployment process"
    
    # Parse command line arguments
    SKIP_BACKUP=false
    SKIP_VERIFICATION=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-verification)
                SKIP_VERIFICATION=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    # Execute deployment steps
    check_prerequisites
    setup_directories
    
    if [ "$SKIP_BACKUP" = false ]; then
        backup_data
    fi
    
    deploy
    run_migrations
    
    if [ "$SKIP_VERIFICATION" = false ]; then
        verify_deployment
    fi
    
    cleanup
    optimize_performance
    
    log "🎉 LinkedIn Posts API deployment completed successfully!"
    log "API is available at: http://localhost"
    log "Prometheus: http://localhost:9090"
    log "Grafana: http://localhost:3000"
    log "Kibana: http://localhost:5601"
    log "Flower: http://localhost:5555"
    
    # Display service status
    echo ""
    log "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
}

# Handle script interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main "$@" 