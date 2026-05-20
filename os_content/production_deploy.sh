#!/bin/bash

# Production Deployment Script for OS Content System
# Complete production setup with monitoring, logging, and backup

set -e

# Configuration
PROJECT_NAME="os-content-production"
COMPOSE_FILE="production_compose.yml"
ENVIRONMENT="production"
DOMAIN="os-content.example.com"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    print_status "Checking production prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Check available disk space (at least 10GB)
    DISK_SPACE=$(df . | awk 'NR==2 {print $4}')
    if [ "$DISK_SPACE" -lt 10485760 ]; then
        print_error "Insufficient disk space. Need at least 10GB free"
        exit 1
    fi
    
    # Check available memory (at least 4GB)
    MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [ "$MEMORY" -lt 4096 ]; then
        print_warning "Low memory detected. Recommended: 8GB+"
    fi
    
    print_success "Prerequisites check passed"
}

# Create production directories
create_directories() {
    print_status "Creating production directories..."
    
    mkdir -p {logs,data,cache,models,backup,ssl,grafana/{dashboards,datasources}}
    
    # Set proper permissions
    chmod 755 logs data cache models backup
    chmod 700 ssl
    
    print_success "Directories created"
}

# Generate SSL certificates
generate_ssl() {
    print_status "Generating SSL certificates..."
    
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        mkdir -p ssl
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/key.pem \
            -out ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"
        
        print_success "SSL certificates generated"
    else
        print_warning "SSL certificates already exist"
    fi
}

# Create configuration files
create_configs() {
    print_status "Creating configuration files..."
    
    # Redis configuration
    cat > redis.conf << EOF
# Redis production configuration
bind 0.0.0.0
port 6379
timeout 300
tcp-keepalive 60
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
EOF

    # Nginx configuration
    cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream os_content_backend {
        server os-content-app:8000;
    }
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=login:10m rate=1r/s;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    server {
        listen 80;
        server_name $DOMAIN;
        return 301 https://\$server_name\$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name $DOMAIN;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://os_content_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # Health check
        location /health {
            proxy_pass http://os_content_backend;
            access_log off;
        }
        
        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Default
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
  - job_name: 'os-content-app'
    static_configs:
      - targets: ['os-content-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
EOF

    # Filebeat configuration
    cat > filebeat.yml << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/os-content/*.log
  fields:
    service: os-content
  fields_under_root: true

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

    print_success "Configuration files created"
}

# Database initialization
init_database() {
    print_status "Creating database initialization script..."
    
    cat > init.sql << EOF
-- OS Content Database Initialization
CREATE DATABASE IF NOT EXISTS os_content;
USE os_content;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Videos table
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    prompt TEXT NOT NULL,
    duration INTEGER NOT NULL,
    output_path VARCHAR(500) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    metadata JSONB
);

-- NLP analysis table
CREATE TABLE IF NOT EXISTS nlp_analyses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    text TEXT NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cache statistics table
CREATE TABLE IF NOT EXISTS cache_stats (
    id SERIAL PRIMARY KEY,
    cache_type VARCHAR(50) NOT NULL,
    hits BIGINT DEFAULT 0,
    misses BIGINT DEFAULT 0,
    size_bytes BIGINT DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    component VARCHAR(50) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_videos_user_id ON videos(user_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_nlp_analyses_user_id ON nlp_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_component ON performance_metrics(component);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at);

-- Insert initial data
INSERT INTO users (username, email, password_hash) VALUES 
('admin', 'admin@os-content.com', 'hashed_password_here')
ON CONFLICT (username) DO NOTHING;
EOF

    print_success "Database initialization script created"
}

# Deploy the system
deploy_system() {
    print_status "Deploying OS Content Production System..."
    
    # Build and start services
    docker-compose -f $COMPOSE_FILE build --no-cache
    
    # Start services in order
    docker-compose -f $COMPOSE_FILE up -d postgres redis
    sleep 10
    
    docker-compose -f $COMPOSE_FILE up -d elasticsearch
    sleep 15
    
    docker-compose -f $COMPOSE_FILE up -d prometheus grafana kibana filebeat
    sleep 10
    
    docker-compose -f $COMPOSE_FILE up -d os-content-app
    sleep 10
    
    docker-compose -f $COMPOSE_FILE up -d nginx
    
    print_success "System deployed successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for database
    until docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U os_content_user -d os_content; do
        sleep 2
    done
    
    # Wait for Redis
    until docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping; do
        sleep 2
    done
    
    # Wait for application
    until curl -f http://localhost:8000/health; do
        sleep 5
    done
    
    # Wait for monitoring
    until curl -f http://localhost:9090/-/healthy; do
        sleep 2
    done
    
    print_success "All services are ready"
}

# Run health checks
run_health_checks() {
    print_status "Running production health checks..."
    
    # Application health
    if curl -f http://localhost:8000/health | grep -q "healthy"; then
        print_success "Application health check passed"
    else
        print_error "Application health check failed"
        return 1
    fi
    
    # Database health
    if docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U os_content_user -d os_content; then
        print_success "Database health check passed"
    else
        print_error "Database health check failed"
        return 1
    fi
    
    # Redis health
    if docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping | grep -q "PONG"; then
        print_success "Redis health check passed"
    else
        print_error "Redis health check failed"
        return 1
    fi
    
    # Monitoring health
    if curl -f http://localhost:9090/-/healthy; then
        print_success "Prometheus health check passed"
    else
        print_error "Prometheus health check failed"
        return 1
    fi
    
    print_success "All health checks passed"
}

# Show deployment information
show_deployment_info() {
    print_success "Production deployment completed successfully!"
    echo ""
    echo "🌐 Services:"
    echo "   Application: https://$DOMAIN"
    echo "   API: https://$DOMAIN/api/"
    echo "   Health: https://$DOMAIN/health"
    echo "   Grafana: http://localhost:3000 (admin/admin)"
    echo "   Prometheus: http://localhost:9090"
    echo "   Kibana: http://localhost:5601"
    echo ""
    echo "🗄️  Databases:"
    echo "   PostgreSQL: localhost:5432"
    echo "   Redis: localhost:6379"
    echo "   Elasticsearch: localhost:9200"
    echo ""
    echo "🔧 Management:"
    echo "   View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "   Stop system: docker-compose -f $COMPOSE_FILE down"
    echo "   Restart: docker-compose -f $COMPOSE_FILE restart"
    echo "   Backup: docker-compose -f $COMPOSE_FILE run backup"
    echo ""
    echo "📊 Monitoring:"
    echo "   System metrics: http://localhost:3000"
    echo "   Application logs: http://localhost:5601"
    echo "   Performance: http://localhost:9090"
    echo ""
}

# Backup function
backup_system() {
    print_status "Creating system backup..."
    
    BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Database backup
    docker-compose -f $COMPOSE_FILE exec -T postgres pg_dump -U os_content_user os_content > "$BACKUP_DIR/database.sql"
    
    # Redis backup
    docker-compose -f $COMPOSE_FILE exec -T redis redis-cli --rdb "$BACKUP_DIR/redis.rdb"
    
    # Configuration backup
    cp {redis.conf,nginx.conf,prometheus.yml,filebeat.yml} "$BACKUP_DIR/"
    
    # Compress backup
    tar -czf "$BACKUP_DIR.tar.gz" -C backup "$(basename $BACKUP_DIR)"
    rm -rf "$BACKUP_DIR"
    
    print_success "Backup created: $BACKUP_DIR.tar.gz"
}

# Main deployment function
deploy() {
    print_status "Starting production deployment..."
    
    check_prerequisites
    create_directories
    generate_ssl
    create_configs
    init_database
    deploy_system
    wait_for_services
    run_health_checks
    show_deployment_info
}

# Cleanup function
cleanup() {
    print_status "Cleaning up production deployment..."
    
    docker-compose -f $COMPOSE_FILE down -v
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "cleanup")
        cleanup
        ;;
    "backup")
        backup_system
        ;;
    "logs")
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
    "status")
        docker-compose -f $COMPOSE_FILE ps
        ;;
    "restart")
        docker-compose -f $COMPOSE_FILE restart
        ;;
    "stop")
        docker-compose -f $COMPOSE_FILE stop
        ;;
    "start")
        docker-compose -f $COMPOSE_FILE start
        ;;
    "update")
        print_status "Updating system..."
        docker-compose -f $COMPOSE_FILE pull
        docker-compose -f $COMPOSE_FILE up -d
        print_success "System updated"
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|backup|logs|status|restart|stop|start|update}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the complete production system"
        echo "  cleanup  - Remove all containers and volumes"
        echo "  backup   - Create system backup"
        echo "  logs     - Show all service logs"
        echo "  status   - Show system status"
        echo "  restart  - Restart all services"
        echo "  stop     - Stop all services"
        echo "  start    - Start all services"
        echo "  update   - Update and restart services"
        exit 1
        ;;
esac 