#!/bin/bash

# 🚀 OPTIMIZED Blaze AI Deployment Script
# Performance-optimized deployment with multiple options

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="blaze-ai-optimized"
DOCKER_COMPOSE_FILE="docker-compose.optimized.yml"
CONFIG_FILE="config-optimized.yaml"

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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
}

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available memory
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 4096 ]; then
        warn "Available memory is less than 4GB. Performance may be limited."
    fi
    
    # Check available disk space
    local available_disk=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$available_disk" -lt 10 ]; then
        warn "Available disk space is less than 10GB. Consider freeing up space."
    fi
    
    log "System requirements check completed"
}

# Check configuration files
check_configuration() {
    log "Checking configuration files..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        error "Configuration file $CONFIG_FILE not found"
        exit 1
    fi
    
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error "Docker Compose file $DOCKER_COMPOSE_FILE not found"
        exit 1
    fi
    
    log "Configuration files check completed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p logs cache models data monitoring/grafana/dashboards monitoring/grafana/datasources nginx/conf.d nginx/ssl redis postgres
    
    log "Directories created successfully"
}

# Generate configuration files
generate_configs() {
    log "Generating configuration files..."
    
    # Generate Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'blaze-ai'
    static_configs:
      - targets: ['blaze-ai:9090']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: /metrics
    scrape_interval: 30s
EOF

    # Generate Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Generate Nginx configuration
    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 100;
    
    # Gzip compression
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
    
    # Upstream servers
    upstream blaze_ai_backend {
        server blaze-ai:8000;
        keepalive 32;
    }
    
    # Main server block
    server {
        listen 80;
        server_name localhost;
        
        # Health check
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://blaze_ai_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            proxy_buffering off;
        }
        
        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Default location
        location / {
            proxy_pass http://blaze_ai_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

    # Generate Redis configuration
    cat > redis/redis.conf << EOF
# Redis configuration for Blaze AI
bind 0.0.0.0
port 6379
timeout 0
tcp-keepalive 300
tcp-backlog 511

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Performance
tcp-keepalive 300
tcp-backlog 511
databases 16

# Security
protected-mode no
EOF

    # Generate PostgreSQL initialization script
    cat > postgres/init.sql << EOF
-- PostgreSQL initialization for Blaze AI
CREATE DATABASE blaze_ai;
\c blaze_ai;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create tables (basic structure)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    permissions TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE blaze_ai TO blazeai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO blazeai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO blazeai;
EOF

    log "Configuration files generated successfully"
}

# Deploy services
deploy_services() {
    local profile="$1"
    
    log "Deploying services with profile: $profile"
    
    if [ "$profile" = "full" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" --profile full up -d
    elif [ "$profile" = "gpu" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" --profile gpu up -d
    elif [ "$profile" = "development" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" --profile development up -d
    else
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    fi
    
    log "Services deployment initiated"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log "Main service is ready!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Service did not become ready within the expected time"
            exit 1
        fi
        
        info "Waiting for service to be ready... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    log "All services are ready!"
}

# Show service status
show_status() {
    log "Service status:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo
    log "Service logs (last 10 lines):"
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=10
}

# Show access information
show_access_info() {
    echo
    log "🚀 Blaze AI Optimized is now running!"
    echo
    echo -e "${CYAN}Access URLs:${NC}"
    echo -e "  🌐 Main API:     ${GREEN}http://localhost:8000${NC}"
    echo -e "  📊 API Docs:     ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  📈 Metrics:      ${GREEN}http://localhost:9090${NC}"
    echo -e "  📊 Grafana:      ${GREEN}http://localhost:3000${NC}"
    echo -e "  🔍 Elasticsearch: ${GREEN}http://localhost:9200${NC}"
    echo -e "  🗄️  PostgreSQL:   ${GREEN}localhost:5432${NC}"
    echo -e "  🗃️  Redis:        ${GREEN}localhost:6379${NC}"
    echo
    echo -e "${CYAN}Default Credentials:${NC}"
    echo -e "  📊 Grafana:      ${YELLOW}admin / admin_change_in_production${NC}"
    echo -e "  🗄️  PostgreSQL:   ${YELLOW}blazeai / blazeai_password_change_in_production${NC}"
    echo
    echo -e "${YELLOW}⚠️  IMPORTANT: Change default passwords in production!${NC}"
}

# Main deployment function
main() {
    local profile="${1:-default}"
    
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🚀 BLAZE AI OPTIMIZED                    ║"
    echo "║                     DEPLOYMENT SCRIPT                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "Starting optimized deployment with profile: $profile"
    
    # Pre-deployment checks
    check_root
    check_system_requirements
    check_configuration
    
    # Setup
    create_directories
    generate_configs
    
    # Deploy
    deploy_services "$profile"
    wait_for_services
    
    # Post-deployment
    show_status
    show_access_info
    
    log "Deployment completed successfully! 🎉"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [PROFILE]"
    echo
    echo "Profiles:"
    echo "  default      - Standard deployment (main + monitoring)"
    echo "  development  - Development environment with hot reload"
    echo "  gpu         - GPU-enabled deployment"
    echo "  full        - Full deployment with all services"
    echo
    echo "Examples:"
    echo "  $0                    # Default deployment"
    echo "  $0 development        # Development environment"
    echo "  $0 gpu               # GPU-enabled deployment"
    echo "  $0 full              # Full deployment"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    default|development|gpu|full)
        main "$1"
        ;;
    "")
        main "default"
        ;;
    *)
        error "Invalid profile: $1"
        show_usage
        exit 1
        ;;
esac

