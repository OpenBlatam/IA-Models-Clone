#!/bin/bash

# Ultra-Optimized SEO Service Deployment Script v8
# Maximum performance deployment with comprehensive setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ultra-seo-service-v8"
DOCKER_COMPOSE_FILE="production/docker-compose.ultra_optimized_v8.yml"
ENV_FILE=".env.production"
BACKUP_DIR="backups"
LOGS_DIR="logs"
CACHE_DIR="cache"
TEMP_DIR="temp"

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
    if [ "$MEMORY_GB" -lt 8 ]; then
        log_warning "Recommended minimum 8GB RAM, found ${MEMORY_GB}GB"
    fi
    
    # Check available disk space
    DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$DISK_GB" -lt 20 ]; then
        log_warning "Recommended minimum 20GB free space, found ${DISK_GB}GB"
    fi
    
    log_success "System requirements check completed"
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$CACHE_DIR"
    mkdir -p "$TEMP_DIR"
    mkdir -p "production/ssl"
    mkdir -p "production/grafana/dashboards"
    mkdir -p "production/grafana/datasources"
    
    log_success "Directories created"
}

generate_ssl_certificates() {
    log_info "Generating SSL certificates..."
    
    if [ ! -f "production/ssl/cert.pem" ] || [ ! -f "production/ssl/key.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout production/ssl/key.pem -out production/ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

create_environment_file() {
    log_info "Creating environment file..."
    
    cat > "$ENV_FILE" << EOF
# Ultra-Optimized SEO Service Environment v8
# Generated on $(date)

# Service Configuration
LOG_LEVEL=INFO
DEBUG=false
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Performance Configuration
MAX_CONNECTIONS=200
CACHE_TTL=3600
MAX_CACHE_SIZE=10000
PREFERRED_PARSER=selectolax
COMPRESSION_ENABLED=true

# Database Configuration
POSTGRES_PASSWORD=ultra_seo_postgres_v8_$(openssl rand -hex 8)
POSTGRES_DB=seo_analysis
POSTGRES_USER=seo_user

# Redis Configuration
REDIS_PASSWORD=ultra_seo_redis_v8_$(openssl rand -hex 8)
REDIS_URL=redis://redis:6379/0

# Monitoring Configuration
GRAFANA_PASSWORD=ultra_seo_grafana_v8_$(openssl rand -hex 8)
PROMETHEUS_MULTIPROC_DIR=/tmp

# Security Configuration
SSL_KEYFILE=/etc/nginx/ssl/key.pem
SSL_CERTFILE=/etc/nginx/ssl/cert.pem
SSL_CA_CERTS=/etc/nginx/ssl/cert.pem

# Backup Configuration
BACKUP_RETENTION_DAYS=30
BACKUP_SCHEDULE="0 2 * * *"

# Performance Tuning
PYTHONOPTIMIZE=2
PYTHONHASHSEED=random
UVLOOP_POLICY=asyncio.WindowsProactorEventLoopPolicy
EOF

    log_success "Environment file created: $ENV_FILE"
}

configure_redis() {
    log_info "Configuring Redis..."
    
    cat > "production/redis.ultra_optimized.conf" << EOF
# Ultra-Optimized Redis Configuration v8
# Maximum performance Redis settings

# Memory Management
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite yes
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 128mb

# Performance
tcp-keepalive 300
tcp-backlog 1024
timeout 0
databases 32
hz 10

# Security
requirepass \${REDIS_PASSWORD}
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""

# Logging
loglevel notice
logfile ""

# Network
bind 0.0.0.0
port 6379
protected-mode no

# Advanced Performance
tcp-nodelay yes
tcp-keepalive 300
repl-backlog-size 1mb
repl-backlog-ttl 3600
EOF

    log_success "Redis configuration created"
}

configure_postgres() {
    log_info "Configuring PostgreSQL..."
    
    cat > "production/postgres.ultra_optimized.conf" << EOF
# Ultra-Optimized PostgreSQL Configuration v8
# Maximum performance PostgreSQL settings

# Memory Configuration
shared_buffers = 512MB
effective_cache_size = 1GB
maintenance_work_mem = 256MB
work_mem = 4MB

# Checkpoint Configuration
checkpoint_completion_target = 0.9
wal_buffers = 16MB
checkpoint_segments = 32

# Query Planner
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# Connection Settings
max_connections = 200
superuser_reserved_connections = 3
unix_socket_directories = '/var/run/postgresql'

# Performance Settings
synchronous_commit = off
wal_sync_method = fdatasync
full_page_writes = off
wal_compression = on
EOF

    log_success "PostgreSQL configuration created"
}

configure_nginx() {
    log_info "Configuring Nginx..."
    
    cat > "production/nginx.ultra_optimized.conf" << EOF
# Ultra-Optimized Nginx Configuration v8
# Maximum performance Nginx settings

user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 65535;
    use epoll;
    multi_accept on;
    accept_mutex off;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format ultra_optimized '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                              '\$status \$body_bytes_sent "\$http_referer" '
                              '"\$http_user_agent" "\$http_x_forwarded_for" '
                              '\$request_time \$upstream_response_time';

    access_log /var/log/nginx/access.log ultra_optimized;

    # Performance Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;
    client_max_body_size 10M;
    client_body_buffer_size 128k;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    output_buffers 1 32k;
    postpone_output 1460;

    # Gzip Compression
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

    # Brotli Compression
    brotli on;
    brotli_comp_level 6;
    brotli_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Rate Limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=20r/s;
    limit_req_zone \$binary_remote_addr zone=burst:10m rate=50r/s;

    # Upstream Configuration
    upstream seo_backend {
        least_conn;
        server seo-service:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # Cache Configuration
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=seo_cache:20m max_size=2g inactive=60m;
    proxy_cache_key "\$scheme\$request_method\$host\$request_uri";
    proxy_cache_valid 200 302 10m;
    proxy_cache_valid 404 1m;
    proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
    proxy_cache_lock on;
    proxy_cache_lock_timeout 5s;

    server {
        listen 80;
        server_name _;
        return 301 https://\$host\$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name _;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin";

        # Rate Limiting
        limit_req zone=api burst=50 nodelay;

        # Proxy Configuration
        location / {
            proxy_pass http://seo_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_cache seo_cache;
            proxy_cache_valid 200 2h;
            proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
            proxy_cache_lock on;
            proxy_read_timeout 30s;
            proxy_connect_timeout 5s;
            proxy_send_timeout 30s;
        }

        # Health Check
        location /health {
            proxy_pass http://seo_backend/health;
            access_log off;
        }

        # Metrics
        location /metrics {
            proxy_pass http://seo_backend/metrics;
            access_log off;
        }
    }
}
EOF

    log_success "Nginx configuration created"
}

configure_prometheus() {
    log_info "Configuring Prometheus..."
    
    cat > "production/prometheus.ultra_optimized.yml" << EOF
# Ultra-Optimized Prometheus Configuration v8
# Maximum performance monitoring settings

global:
  scrape_interval: 10s
  evaluation_interval: 10s
  external_labels:
    monitor: 'ultra-seo-service-v8'

rule_files:
  - "seo_rules.yml"

scrape_configs:
  - job_name: 'ultra-seo-service'
    static_configs:
      - targets: ['seo-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
    scrape_timeout: 3s
    honor_labels: true
    scheme: http

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: '/nginx_status'
    scrape_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_groups:
  - name: seo_service_alerts
    rules:
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for 5 minutes"

      - alert: HighMemoryUsage
        expr: (process_resident_memory_bytes / container_memory_usage_bytes) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 80% for 5 minutes"

      - alert: ServiceDown
        expr: up{job="ultra-seo-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "SEO service is down"
          description: "The SEO service has been down for more than 1 minute"
EOF

    log_success "Prometheus configuration created"
}

configure_alertmanager() {
    log_info "Configuring Alertmanager..."
    
    cat > "production/alertmanager.ultra_optimized.yml" << EOF
# Ultra-Optimized Alertmanager Configuration v8
# Maximum performance alerting settings

global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR_SLACK_WEBHOOK'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'slack'
      continue: true

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://127.0.0.1:5001/'

  - name: 'slack'
    slack_configs:
      - channel: '#alerts'
        title: '{{ template "slack.title" . }}'
        text: '{{ template "slack.text" . }}'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

    log_success "Alertmanager configuration created"
}

configure_filebeat() {
    log_info "Configuring Filebeat..."
    
    cat > "production/filebeat.ultra_optimized.yml" << EOF
# Ultra-Optimized Filebeat Configuration v8
# Maximum performance log collection

filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/seo/*.log
  fields:
    service: seo-service
  fields_under_root: true
  multiline.pattern: '^\['
  multiline.negate: true
  multiline.match: after

- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_docker_metadata:
        host: "unix:///var/run/docker.sock"

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
  - add_cloud_metadata: ~
  - add_docker_metadata: ~
  - add_kubernetes_metadata: ~

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  indices:
    - index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"

setup.kibana:
  host: "kibana:5601"

setup.dashboards.enabled: true
setup.template.enabled: true
setup.template.name: "filebeat"
setup.template.pattern: "filebeat-*"
setup.template.overwrite: true
EOF

    log_success "Filebeat configuration created"
}

setup_grafana_dashboards() {
    log_info "Setting up Grafana dashboards..."
    
    # Create datasource configuration
    cat > "production/grafana/datasources/datasource.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Create dashboard configuration
    cat > "production/grafana/dashboards/dashboard.yml" << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    log_success "Grafana configuration created"
}

deploy_services() {
    log_info "Deploying ultra-optimized services..."
    
    # Load environment variables
    source "$ENV_FILE"
    
    # Build and start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build
    
    log_success "Services deployed successfully"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL..."
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U seo_user -d seo_analysis; do
        sleep 5
    done
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    until docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping; do
        sleep 5
    done
    
    # Wait for Elasticsearch
    log_info "Waiting for Elasticsearch..."
    until curl -f http://localhost:9200/_cluster/health; do
        sleep 10
    done
    
    # Wait for SEO service
    log_info "Waiting for SEO service..."
    until curl -f http://localhost:8000/health; do
        sleep 10
    done
    
    log_success "All services are ready"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Check service health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "SEO service is healthy"
    else
        log_error "SEO service health check failed"
        exit 1
    fi
    
    # Check monitoring stack
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_success "Prometheus is healthy"
    else
        log_warning "Prometheus health check failed"
    fi
    
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        log_success "Grafana is healthy"
    else
        log_warning "Grafana health check failed"
    fi
    
    log_success "Health checks completed"
}

setup_backup_cron() {
    log_info "Setting up backup cron job..."
    
    # Create backup script
    cat > "backup.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker-compose -f production/docker-compose.ultra_optimized_v8.yml exec -T postgres pg_dump -U seo_user -d seo_analysis > backups/postgres_$(date +%Y%m%d_%H%M%S).sql
docker-compose -f production/docker-compose.ultra_optimized_v8.yml exec -T redis redis-cli BGSAVE
echo "Backup completed at $(date)" >> backups/backup.log
EOF

    chmod +x backup.sh
    
    # Add to crontab (daily at 2 AM)
    (crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/backup.sh") | crontab -
    
    log_success "Backup cron job configured"
}

display_info() {
    log_success "Ultra-Optimized SEO Service v8 deployment completed!"
    echo
    echo "Service URLs:"
    echo "  SEO Service:     http://localhost:8000"
    echo "  API Docs:        http://localhost:8000/docs"
    echo "  Health Check:    http://localhost:8000/health"
    echo "  Metrics:         http://localhost:8000/metrics"
    echo "  Grafana:         http://localhost:3000"
    echo "  Prometheus:      http://localhost:9090"
    echo "  Kibana:          http://localhost:5601"
    echo "  Alertmanager:    http://localhost:9093"
    echo
    echo "Default credentials:"
    echo "  Grafana:         admin / $(grep GRAFANA_PASSWORD "$ENV_FILE" | cut -d'=' -f2)"
    echo "  PostgreSQL:      seo_user / $(grep POSTGRES_PASSWORD "$ENV_FILE" | cut -d'=' -f2)"
    echo "  Redis:           $(grep REDIS_PASSWORD "$ENV_FILE" | cut -d'=' -f2)"
    echo
    echo "Useful commands:"
    echo "  View logs:       docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
    echo "  Stop services:   docker-compose -f $DOCKER_COMPOSE_FILE down"
    echo "  Restart:         docker-compose -f $DOCKER_COMPOSE_FILE restart"
    echo "  Backup:          ./backup.sh"
    echo
}

# Main deployment process
main() {
    log_info "Starting Ultra-Optimized SEO Service v8 deployment..."
    
    check_requirements
    create_directories
    generate_ssl_certificates
    create_environment_file
    configure_redis
    configure_postgres
    configure_nginx
    configure_prometheus
    configure_alertmanager
    configure_filebeat
    setup_grafana_dashboards
    deploy_services
    wait_for_services
    run_health_checks
    setup_backup_cron
    display_info
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@" 