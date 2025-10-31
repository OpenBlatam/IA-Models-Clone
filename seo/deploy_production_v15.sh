#!/bin/bash

# Ultra-Optimized SEO Service v15 - Production Deployment Script
# Complete deployment with monitoring, logging, and security

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="seo-service-v15"
DOCKER_COMPOSE_FILE="docker-compose.production_v15.yml"
ENV_FILE=".env.production_v15"

# Logging
LOG_FILE="deploy_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
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
    if [ "$MEMORY_GB" -lt 8 ]; then
        log_warn "Recommended minimum 8GB RAM, found ${MEMORY_GB}GB"
    fi
    
    # Check available disk space
    DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$DISK_GB" -lt 20 ]; then
        log_warn "Recommended minimum 20GB free space, found ${DISK_GB}GB"
    fi
    
    log_info "Prerequisites check completed"
}

# Create environment file
create_env_file() {
    log_step "Creating environment file..."
    
    cat > "$ENV_FILE" << EOF
# Ultra-Optimized SEO Service v15 - Production Environment
# Generated on $(date)

# Application settings
DEBUG=false
HOST=0.0.0.0
PORT=8000
WORKERS=4
MAX_CONNECTIONS=1000
TIMEOUT=30
RATE_LIMIT=100
CACHE_TTL=3600

# Database settings
REDIS_URL=redis://redis:6379
MONGO_URL=mongodb://admin:secure_password_2024@mongo:27017
MONGO_DATABASE=seo_service

# Security settings
JWT_SECRET=$(openssl rand -hex 32)
BCRYPT_ROUNDS=12

# Monitoring settings
SENTRY_DSN=
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Logging settings
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance settings
UVICORN_WORKERS=4
UVICORN_LOOP=uvloop
UVICORN_HTTP=httptools

# SSL/TLS settings (if using HTTPS)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Backup settings
BACKUP_RETENTION_DAYS=30
BACKUP_SCHEDULE="0 2 * * *"

# Rate limiting
RATE_LIMIT_WINDOW=60
RATE_LIMIT_MAX_REQUESTS=100

# Cache settings
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5

# MongoDB settings
MONGO_MAX_POOL_SIZE=100
MONGO_MIN_POOL_SIZE=10
MONGO_MAX_IDLE_TIME_MS=30000

# Nginx settings
NGINX_WORKER_PROCESSES=auto
NGINX_WORKER_CONNECTIONS=1024
NGINX_KEEPALIVE_TIMEOUT=65
NGINX_CLIENT_MAX_BODY_SIZE=10M

# Prometheus settings
PROMETHEUS_RETENTION_TIME=200h
PROMETHEUS_STORAGE_PATH=/prometheus

# Grafana settings
GRAFANA_ADMIN_PASSWORD=admin_secure_2024
GRAFANA_ALLOW_SIGN_UP=false

# Elasticsearch settings
ES_JAVA_OPTS=-Xms1g -Xmx1g
ES_DISCOVERY_TYPE=single-node
ES_XPACK_SECURITY_ENABLED=false

# Filebeat settings
FILEBEAT_LOG_LEVEL=info
FILEBEAT_SCAN_FREQUENCY=10s
EOF

    log_info "Environment file created: $ENV_FILE"
}

# Create SSL certificates (self-signed for development)
create_ssl_certificates() {
    log_step "Creating SSL certificates..."
    
    mkdir -p ssl
    
    # Generate self-signed certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/key.pem \
        -out ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    # Set proper permissions
    chmod 600 ssl/key.pem
    chmod 644 ssl/cert.pem
    
    log_info "SSL certificates created"
}

# Create configuration files
create_config_files() {
    log_step "Creating configuration files..."
    
    # Redis configuration
    cat > redis.optimized.conf << EOF
# Redis optimized configuration for SEO service
bind 0.0.0.0
port 6379
timeout 0
tcp-keepalive 300
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
logfile ""
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data
maxmemory 1gb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes
EOF

    # Nginx configuration
    cat > nginx.optimized.conf << EOF
user nginx;
worker_processes auto;
worker_cpu_affinity auto;
worker_rlimit_nofile 65535;

error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                    '\$status \$body_bytes_sent "\$http_referer" '
                    '"\$http_user_agent" "\$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 100;
    client_max_body_size 10M;
    client_body_buffer_size 128k;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=login:10m rate=1r/s;

    # Upstream
    upstream seo_backend {
        server seo-service:8000;
        keepalive 32;
    }

    # HTTP server
    server {
        listen 80;
        server_name _;
        
        # Redirect to HTTPS
        return 301 https://\$server_name\$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name _;

        # SSL
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://seo_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_cache_bypass \$http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # Health check
        location /health {
            proxy_pass http://seo_backend/health;
            access_log off;
        }

        # Metrics
        location /metrics {
            proxy_pass http://seo_backend/metrics;
            access_log off;
        }

        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Root
        location / {
            return 404;
        }
    }
}
EOF

    # Prometheus configuration
    cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'seo-service'
    static_configs:
      - targets: ['seo-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongo:27017']
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
  json.message_key: log

- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'

processors:
- add_docker_metadata:
    host: "unix:///var/run/docker.sock"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  indices:
    - index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"

setup.kibana:
  host: "kibana:5601"

setup.dashboards.enabled: true
setup.template.enabled: true
EOF

    log_info "Configuration files created"
}

# Create directories
create_directories() {
    log_step "Creating directories..."
    
    mkdir -p {logs,cache,data,ssl,grafana-dashboards,grafana-datasources}
    
    log_info "Directories created"
}

# Build and deploy
deploy_services() {
    log_step "Building and deploying services..."
    
    # Pull latest images
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # Build services
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    # Deploy services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_info "Services deployed"
}

# Wait for services to be ready
wait_for_services() {
    log_step "Waiting for services to be ready..."
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; do
        sleep 2
    done
    
    # Wait for MongoDB
    log_info "Waiting for MongoDB..."
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T mongo mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
        sleep 2
    done
    
    # Wait for SEO service
    log_info "Waiting for SEO service..."
    until curl -f http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
    done
    
    # Wait for Prometheus
    log_info "Waiting for Prometheus..."
    until curl -f http://localhost:9090/-/ready > /dev/null 2>&1; do
        sleep 5
    done
    
    # Wait for Grafana
    log_info "Waiting for Grafana..."
    until curl -f http://localhost:3000/api/health > /dev/null 2>&1; do
        sleep 5
    done
    
    # Wait for Elasticsearch
    log_info "Waiting for Elasticsearch..."
    until curl -f http://localhost:9200/_cluster/health > /dev/null 2>&1; do
        sleep 5
    done
    
    # Wait for Kibana
    log_info "Waiting for Kibana..."
    until curl -f http://localhost:5601/api/status > /dev/null 2>&1; do
        sleep 5
    done
    
    log_info "All services are ready"
}

# Run health checks
run_health_checks() {
    log_step "Running health checks..."
    
    # Check SEO service
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "SEO service is healthy"
    else
        log_error "SEO service health check failed"
        return 1
    fi
    
    # Check Redis
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
        log_info "Redis is healthy"
    else
        log_error "Redis health check failed"
        return 1
    fi
    
    # Check MongoDB
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T mongo mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        log_info "MongoDB is healthy"
    else
        log_error "MongoDB health check failed"
        return 1
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/ready > /dev/null 2>&1; then
        log_info "Prometheus is healthy"
    else
        log_error "Prometheus health check failed"
        return 1
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        log_info "Grafana is healthy"
    else
        log_error "Grafana health check failed"
        return 1
    fi
    
    # Check Elasticsearch
    if curl -f http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        log_info "Elasticsearch is healthy"
    else
        log_error "Elasticsearch health check failed"
        return 1
    fi
    
    # Check Kibana
    if curl -f http://localhost:5601/api/status > /dev/null 2>&1; then
        log_info "Kibana is healthy"
    else
        log_error "Kibana health check failed"
        return 1
    fi
    
    log_info "All health checks passed"
}

# Show service status
show_status() {
    log_step "Service status:"
    
    echo
    echo "=== Service URLs ==="
    echo "SEO Service API:     http://localhost:8000"
    echo "SEO Service Docs:    http://localhost:8000/docs"
    echo "Prometheus:          http://localhost:9090"
    echo "Grafana:             http://localhost:3000 (admin/admin_secure_2024)"
    echo "Elasticsearch:       http://localhost:9200"
    echo "Kibana:              http://localhost:5601"
    echo "Redis:               localhost:6379"
    echo "MongoDB:             localhost:27017"
    echo
    
    echo "=== Container Status ==="
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo
    echo "=== Resource Usage ==="
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Main deployment function
main() {
    log_info "Starting Ultra-Optimized SEO Service v15 deployment..."
    
    check_prerequisites
    create_env_file
    create_ssl_certificates
    create_config_files
    create_directories
    deploy_services
    wait_for_services
    run_health_checks
    show_status
    
    log_info "Deployment completed successfully!"
    log_info "Log file: $LOG_FILE"
}

# Run main function
main "$@" 