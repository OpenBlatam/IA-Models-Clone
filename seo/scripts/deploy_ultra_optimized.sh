#!/bin/bash

# Production Deployment Script for Ultra-Optimized SEO Service
# Version: 2.0.0
# Environment: Production

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="seo-service-ultra-optimized"
DOCKER_COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE=".env.production"
BACKUP_DIR="/backups"
LOG_DIR="/var/log/seo-service"
CONFIG_DIR="/etc/seo-service"

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
    
    # Check available disk space (at least 10GB)
    DISK_SPACE=$(df / | awk 'NR==2 {print $4}')
    if [ "$DISK_SPACE" -lt 10485760 ]; then
        log_error "Insufficient disk space. Need at least 10GB free"
        exit 1
    fi
    
    # Check available memory (at least 4GB)
    MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [ "$MEMORY" -lt 4096 ]; then
        log_warning "Low memory detected. Recommended: 4GB+"
    fi
    
    log_success "System requirements check passed"
}

setup_directories() {
    log_info "Setting up directories..."
    
    # Create necessary directories
    sudo mkdir -p "$BACKUP_DIR"
    sudo mkdir -p "$LOG_DIR"
    sudo mkdir -p "$CONFIG_DIR"
    sudo mkdir -p "/var/lib/seo-service"
    sudo mkdir -p "/var/lib/seo-service/cache"
    sudo mkdir -p "/var/lib/seo-service/screenshots"
    
    # Set permissions
    sudo chown -R $USER:$USER "$BACKUP_DIR"
    sudo chown -R $USER:$USER "$LOG_DIR"
    sudo chown -R $USER:$USER "$CONFIG_DIR"
    sudo chown -R $USER:$USER "/var/lib/seo-service"
    
    log_success "Directories created successfully"
}

setup_environment() {
    log_info "Setting up environment variables..."
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        log_warning "Environment file not found. Creating from template..."
        cp .env.example "$ENV_FILE"
    fi
    
    # Load environment variables
    source "$ENV_FILE"
    
    # Validate required variables
    REQUIRED_VARS=(
        "OPENAI_API_KEY"
        "REDIS_PASSWORD"
        "DB_PASSWORD"
        "SECRET_KEY"
        "API_KEY"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Environment variables validated"
}

setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    # Check if certificates exist
    if [ ! -f "/etc/ssl/certs/seo-service.crt" ] || [ ! -f "/etc/ssl/private/seo-service.key" ]; then
        log_warning "SSL certificates not found. Generating self-signed certificates..."
        
        # Generate self-signed certificate
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout /etc/ssl/private/seo-service.key \
            -out /etc/ssl/certs/seo-service.crt \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=seo-service.local"
        
        # Set permissions
        sudo chmod 600 /etc/ssl/private/seo-service.key
        sudo chmod 644 /etc/ssl/certs/seo-service.crt
    fi
    
    log_success "SSL certificates configured"
}

setup_redis() {
    log_info "Setting up Redis configuration..."
    
    # Create Redis configuration
    cat > redis.production.conf << EOF
# Redis Production Configuration
bind 0.0.0.0
port 6379
requirepass $REDIS_PASSWORD
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
tcp-keepalive 300
timeout 0
tcp-backlog 511
databases 16
EOF
    
    log_success "Redis configuration created"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create Prometheus configuration
    cat > prometheus.production.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert.rules"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'seo-service'
    static_configs:
      - targets: ['seo-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
    
    # Create Grafana configuration
    cat > grafana.ini << EOF
[server]
http_port = 3000
domain = localhost
root_url = %(protocol)s://%(domain)s:%(http_port)s/

[database]
type = sqlite3
path = /var/lib/grafana/grafana.db

[security]
admin_user = admin
admin_password = admin
secret_key = $(openssl rand -hex 32)

[users]
allow_sign_up = false

[auth.anonymous]
enabled = false
EOF
    
    log_success "Monitoring configuration created"
}

setup_logging() {
    log_info "Setting up logging configuration..."
    
    # Create logrotate configuration
    sudo tee /etc/logrotate.d/seo-service > /dev/null << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        docker-compose -f $DOCKER_COMPOSE_FILE restart seo-api
    endscript
}
EOF
    
    log_success "Logging configuration created"
}

backup_existing() {
    log_info "Creating backup of existing deployment..."
    
    BACKUP_NAME="backup-$(date +%Y%m%d-%H%M%S)"
    BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup configuration files
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        cp "$DOCKER_COMPOSE_FILE" "$BACKUP_PATH/"
    fi
    
    if [ -f "$ENV_FILE" ]; then
        cp "$ENV_FILE" "$BACKUP_PATH/"
    fi
    
    # Backup data volumes
    if docker volume ls | grep -q seo-service; then
        docker run --rm -v seo-service-data:/data -v "$BACKUP_PATH":/backup alpine tar czf /backup/data.tar.gz -C /data .
    fi
    
    log_success "Backup created: $BACKUP_PATH"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build main service image
    docker build -f Dockerfile.production -t seo-service:latest .
    
    # Build additional images if needed
    if [ -f "Dockerfile.nginx" ]; then
        docker build -f Dockerfile.nginx -t seo-nginx:latest .
    fi
    
    log_success "Docker images built successfully"
}

deploy_services() {
    log_info "Deploying services..."
    
    # Stop existing services
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        log_info "Stopping existing services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
    fi
    
    # Start services
    log_info "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    log_success "Services deployed successfully"
}

run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for database to be ready
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U postgres; do
        log_info "Waiting for database..."
        sleep 5
    done
    
    # Run migrations
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T seo-api python -m alembic upgrade head
    
    log_success "Database migrations completed"
}

health_check() {
    log_info "Performing health checks..."
    
    # Check if services are running
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        log_error "Services are not running"
        return 1
    fi
    
    # Check API health
    for i in {1..30}; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            log_success "API health check passed"
            break
        fi
        
        if [ $i -eq 30 ]; then
            log_error "API health check failed after 30 attempts"
            return 1
        fi
        
        log_info "Waiting for API to be ready... (attempt $i/30)"
        sleep 10
    done
    
    # Check Redis
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli -a "$REDIS_PASSWORD" ping | grep -q "PONG"; then
        log_error "Redis health check failed"
        return 1
    fi
    
    # Check database
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U postgres; then
        log_error "Database health check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

setup_monitoring_dashboards() {
    log_info "Setting up monitoring dashboards..."
    
    # Wait for Grafana to be ready
    until curl -f -s http://localhost:3000/api/health > /dev/null; do
        log_info "Waiting for Grafana..."
        sleep 5
    done
    
    # Import dashboards
    if [ -f "grafana-dashboards/seo-dashboard.json" ]; then
        curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
            -H "Content-Type: application/json" \
            -d @grafana-dashboards/seo-dashboard.json
    fi
    
    log_success "Monitoring dashboards configured"
}

setup_backup_schedule() {
    log_info "Setting up backup schedule..."
    
    # Create backup script
    cat > /usr/local/bin/seo-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups"
SERVICE_NAME="seo-service-ultra-optimized"
DATE=$(date +%Y%m%d-%H%M%S)

# Create backup
docker-compose -f docker-compose.production.yml exec -T postgres pg_dump -U postgres seo_db > "$BACKUP_DIR/db-backup-$DATE.sql"
docker run --rm -v seo-service-data:/data -v "$BACKUP_DIR":/backup alpine tar czf "/backup/data-backup-$DATE.tar.gz" -C /data .

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
EOF
    
    chmod +x /usr/local/bin/seo-backup.sh
    
    # Add to crontab
    (crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/seo-backup.sh") | crontab -
    
    log_success "Backup schedule configured"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove temporary files
    rm -f redis.production.conf
    rm -f prometheus.production.yml
    rm -f grafana.ini
    
    # Clean up old images
    docker image prune -f
    
    log_success "Cleanup completed"
}

main() {
    log_info "Starting production deployment of Ultra-Optimized SEO Service..."
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        log_error "Please do not run this script as root"
        exit 1
    fi
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-backup       Skip backup of existing deployment"
                echo "  --skip-health-check Skip health checks after deployment"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute deployment steps
    check_requirements
    setup_directories
    setup_environment
    setup_ssl
    setup_redis
    setup_monitoring
    setup_logging
    
    if [ "${SKIP_BACKUP:-false}" != "true" ]; then
        backup_existing
    fi
    
    build_images
    deploy_services
    run_migrations
    
    if [ "${SKIP_HEALTH_CHECK:-false}" != "true" ]; then
        health_check
    fi
    
    setup_monitoring_dashboards
    setup_backup_schedule
    cleanup
    
    log_success "Production deployment completed successfully!"
    log_info "Service URL: https://localhost"
    log_info "API Documentation: https://localhost/docs"
    log_info "Grafana Dashboard: http://localhost:3000"
    log_info "Prometheus: http://localhost:9090"
}

# Run main function
main "$@" 