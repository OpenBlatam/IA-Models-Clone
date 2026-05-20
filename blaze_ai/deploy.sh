#!/bin/bash

# Enhanced Blaze AI Deployment Script
# Automated deployment with all enterprise-grade features

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="blaze-ai-enhanced"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Logging functions
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

# Print header
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  Enhanced Blaze AI Deployment${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "All prerequisites are satisfied!"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration file..."
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Enhanced Blaze AI Environment Configuration
# Generated on $(date)

# Core Configuration
APP_ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)
API_KEY_REQUIRED=true

# External Services (Update with your actual keys)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
STABILITY_API_KEY=your-stability-api-key-here

# Database (Optional)
DATABASE_URL=postgresql://user:password@localhost:5432/blaze_ai

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PASSWORD=admin123

# Feature Flags
ENABLE_ADVANCED_SECURITY=true
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_RATE_LIMITING=true
ENABLE_CIRCUIT_BREAKER=true
ENABLE_ERROR_RECOVERY=true
ENABLE_AUDIT_LOGGING=true
ENABLE_METRICS_EXPORT=true
ENABLE_HEALTH_ENDPOINTS=true
EOF
        
        log_success "Environment file created: $ENV_FILE"
        log_warning "Please update the API keys in $ENV_FILE with your actual values!"
    else
        log_info "Environment file already exists: $ENV_FILE"
    fi
}

# Create monitoring configuration
create_monitoring_config() {
    log_info "Creating monitoring configuration..."
    
    # Create monitoring directory
    mkdir -p monitoring/prometheus
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    
    # Create Prometheus configuration
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
      - targets: ['blaze-ai:8000']
    metrics_path: '/metrics/prometheus'
    scrape_interval: 5s
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    
    # Create Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    # Create basic Grafana dashboard
    cat > monitoring/grafana/dashboards/blaze-ai-dashboard.json << EOF
{
  "dashboard": {
    "id": null,
    "title": "Blaze AI Enhanced Dashboard",
    "tags": ["blaze-ai", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"blaze-ai\"}",
            "legendFormat": "{{job}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF
    
    log_success "Monitoring configuration created!"
}

# Create Nginx configuration
create_nginx_config() {
    log_info "Creating Nginx configuration..."
    
    mkdir -p nginx/ssl
    
    # Create Nginx configuration
    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream blaze_ai_backend {
        server blaze-ai:8000;
    }
    
    upstream grafana_backend {
        server grafana:3000;
    }
    
    upstream prometheus_backend {
        server prometheus:9090;
    }
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=login:10m rate=1r/s;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    server {
        listen 80;
        server_name localhost;
        
        # Redirect to HTTPS
        return 301 https://\$server_name\$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name localhost;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # API endpoints
        location / {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://blaze_ai_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Health check (no rate limiting)
        location /health {
            proxy_pass http://blaze_ai_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Metrics (protected)
        location /metrics {
            auth_basic "Metrics";
            auth_basic_user_file /etc/nginx/.htpasswd;
            proxy_pass http://blaze_ai_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Grafana
        location /grafana/ {
            proxy_pass http://grafana_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Prometheus
        location /prometheus/ {
            auth_basic "Prometheus";
            auth_basic_user_file /etc/nginx/.htpasswd;
            proxy_pass http://prometheus_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF
    
    # Create self-signed SSL certificate for development
    if [ ! -f nginx/ssl/cert.pem ]; then
        log_info "Creating self-signed SSL certificate..."
        openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        log_success "SSL certificate created!"
    fi
    
    log_success "Nginx configuration created!"
}

# Build and start services
deploy_services() {
    log_info "Building and starting services..."
    
    # Build images
    log_info "Building Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    # Start services
    log_info "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_success "Services started successfully!"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for Blaze AI
    log_info "Waiting for Blaze AI service..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "Blaze AI service is ready!"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "Timeout waiting for Blaze AI service"
        exit 1
    fi
    
    # Wait for Redis
    log_info "Waiting for Redis service..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec blaze-ai-redis redis-cli ping &> /dev/null; then
            log_success "Redis service is ready!"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "Timeout waiting for Redis service"
        exit 1
    fi
    
    # Wait for Prometheus
    log_info "Waiting for Prometheus service..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:9091/-/healthy &> /dev/null; then
            log_success "Prometheus service is ready!"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done
    
    if [ $timeout -le 0 ]; then
        log_warning "Timeout waiting for Prometheus service"
    fi
    
    # Wait for Grafana
    log_info "Waiting for Grafana service..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000/api/health &> /dev/null; then
            log_success "Grafana service is ready!"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done
    
    if [ $timeout -le 0 ]; then
        log_warning "Timeout waiting for Grafana service"
    fi
}

# Run tests
run_tests() {
    log_info "Running system tests..."
    
    # Test basic functionality
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Health check passed!"
    else
        log_error "Health check failed!"
        return 1
    fi
    
    # Test metrics endpoint
    if curl -f http://localhost:8000/metrics &> /dev/null; then
        log_success "Metrics endpoint accessible!"
    else
        log_warning "Metrics endpoint not accessible (expected in demo mode)"
    fi
    
    # Test security endpoint
    if curl -f http://localhost:8000/security/status &> /dev/null; then
        log_success "Security endpoint accessible!"
    else
        log_error "Security endpoint failed!"
        return 1
    fi
    
    log_success "All tests passed!"
}

# Show deployment information
show_deployment_info() {
    log_success "Deployment completed successfully!"
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  Deployment Information${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}Service URLs:${NC}"
    echo -e "  🌐 Blaze AI API:     ${GREEN}http://localhost:8000${NC}"
    echo -e "  📊 API Docs:         ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  🏥 Health Check:     ${GREEN}http://localhost:8000/health${NC}"
    echo -e "  📈 Metrics:          ${GREEN}http://localhost:8000/metrics${NC}"
    echo -e "  🔒 Security Status:  ${GREEN}http://localhost:8000/security/status${NC}"
    echo -e "  📊 Grafana:          ${GREEN}http://localhost:3000${NC}"
    echo -e "  📊 Prometheus:       ${GREEN}http://localhost:9091${NC}"
    echo -e "  🗄️  Redis:           ${GREEN}localhost:6379${NC}"
    
    echo -e "\n${BLUE}Default Credentials:${NC}"
    echo -e "  📊 Grafana:          ${GREEN}admin / admin123${NC}"
    
    echo -e "\n${BLUE}Useful Commands:${NC}"
    echo -e "  📋 View logs:        ${GREEN}docker-compose logs -f${NC}"
    echo -e "  🛑 Stop services:    ${GREEN}docker-compose down${NC}"
    echo -e "  🔄 Restart:          ${GREEN}docker-compose restart${NC}"
    echo -e "  🧪 Run tests:        ${GREEN}python test_enhanced_features.py${NC}"
    echo -e "  🎯 Run demo:         ${GREEN}python demo_enhanced_features.py${NC}"
    
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "  1. Update API keys in ${GREEN}$ENV_FILE${NC}"
    echo -e "  2. Configure your domain in Nginx (if using production)"
    echo -e "  3. Set up SSL certificates for production use"
    echo -e "  4. Customize monitoring dashboards in Grafana"
    echo -e "  5. Run the interactive demo: ${GREEN}python demo_enhanced_features.py${NC}"
    
    echo -e "\n${GREEN}🎉 Your enhanced Blaze AI system is ready!${NC}\n"
}

# Main deployment function
main() {
    print_header
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        log_error "Please do not run this script as root"
        exit 1
    fi
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Create configuration files
    create_env_file
    create_monitoring_config
    create_nginx_config
    
    # Deploy services
    deploy_services
    
    # Wait for services
    wait_for_services
    
    # Run tests
    run_tests
    
    # Show deployment information
    show_deployment_info
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --dev          Deploy development environment"
        echo "  --test         Run tests only"
        echo "  --stop         Stop all services"
        echo "  --logs         Show service logs"
        echo "  --status       Show service status"
        ;;
    --dev)
        log_info "Starting development environment..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" --profile dev up -d
        ;;
    --test)
        log_info "Running tests..."
        wait_for_services
        run_tests
        ;;
    --stop)
        log_info "Stopping all services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        ;;
    --logs)
        log_info "Showing service logs..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
        ;;
    --status)
        log_info "Showing service status..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
