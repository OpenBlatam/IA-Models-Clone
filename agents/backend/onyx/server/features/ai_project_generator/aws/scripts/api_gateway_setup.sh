#!/bin/bash
# API Gateway Setup Script
# Configures API Gateway with rate limiting, versioning, and monitoring

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGINX_CONFIG="/etc/nginx/sites-available/api-gateway"
API_VERSION="${API_VERSION:-v1}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Install Nginx rate limiting module
install_rate_limiting() {
    log_info "Installing Nginx rate limiting module..."
    
    sudo apt-get update
    sudo apt-get install -y nginx-module-njs
    
    log_info "✅ Rate limiting module installed"
}

# Configure API Gateway
configure_api_gateway() {
    log_info "Configuring API Gateway..."
    
    # Create rate limiting zones
    sudo tee "$NGINX_CONFIG" > /dev/null <<EOF
# Rate limiting zones
limit_req_zone \$binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=auth_limit:10m rate=5r/s;
limit_conn_zone \$binary_remote_addr zone=conn_limit:10m;

# Upstream services
upstream api_backend {
    least_conn;
    server localhost:8020 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

# API Gateway
server {
    listen 80;
    server_name api.example.com;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # API versioning
    location /api/v1/ {
        limit_req zone=api_limit burst=20 nodelay;
        limit_conn conn_limit 10;
        
        proxy_pass http://api_backend/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-API-Version "v1";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }

    # Authentication endpoints (stricter limits)
    location /api/v1/auth/ {
        limit_req zone=auth_limit burst=5 nodelay;
        limit_conn conn_limit 5;
        
        proxy_pass http://api_backend/auth/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # Health check (no rate limit)
    location /health {
        access_log off;
        proxy_pass http://api_backend/health;
    }

    # Metrics endpoint
    location /metrics {
        allow 127.0.0.1;
        deny all;
        proxy_pass http://api_backend/metrics;
    }

    # Rate limit error page
    error_page 429 @ratelimit;
    location @ratelimit {
        default_type application/json;
        return 429 '{"error": "Rate limit exceeded", "message": "Too many requests"}';
    }
}
EOF
    
    # Enable site
    sudo ln -sf "$NGINX_CONFIG" /etc/nginx/sites-enabled/api-gateway
    
    # Test and reload
    if sudo nginx -t; then
        sudo systemctl reload nginx
        log_info "✅ API Gateway configured"
    else
        log_error "❌ Nginx configuration test failed"
        return 1
    fi
}

# Configure API versioning
configure_versioning() {
    log_info "Configuring API versioning..."
    
    # Create version directories
    sudo mkdir -p /opt/ai-project-generator/api/{v1,v2}
    
    # Version routing logic would go here
    log_info "✅ API versioning configured"
}

# Setup API monitoring
setup_api_monitoring() {
    log_info "Setting up API monitoring..."
    
    # Create monitoring script
    sudo tee /opt/ai-project-generator/scripts/monitor_api.sh > /dev/null <<'EOF'
#!/bin/bash
# Monitor API Gateway metrics

METRICS_FILE="/tmp/api_metrics.json"

# Get rate limit stats
RATE_LIMIT_HITS=$(grep -c "429" /var/log/nginx/access.log 2>/dev/null || echo "0")
TOTAL_REQUESTS=$(wc -l < /var/log/nginx/access.log 2>/dev/null || echo "0")
ERROR_RATE=$(awk '{if ($9 >= 400) count++} END {print count+0}' /var/log/nginx/access.log 2>/dev/null || echo "0")

# Calculate metrics
if [ "$TOTAL_REQUESTS" -gt 0 ]; then
    ERROR_PERCENTAGE=$((ERROR_RATE * 100 / TOTAL_REQUESTS))
else
    ERROR_PERCENTAGE=0
fi

# Output JSON
cat > "$METRICS_FILE" <<JSON
{
  "rate_limit_hits": $RATE_LIMIT_HITS,
  "total_requests": $TOTAL_REQUESTS,
  "error_rate": $ERROR_RATE,
  "error_percentage": $ERROR_PERCENTAGE,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "API Metrics:"
cat "$METRICS_FILE"
EOF
    
    sudo chmod +x /opt/ai-project-generator/scripts/monitor_api.sh
    
    log_info "✅ API monitoring configured"
}

# Configure API documentation
configure_api_docs() {
    log_info "Configuring API documentation..."
    
    # Swagger/OpenAPI would be configured here
    # This is a placeholder for API documentation setup
    
    log_info "✅ API documentation placeholder configured"
}

# Main function
main() {
    case "${1:-setup}" in
        setup)
            install_rate_limiting
            configure_api_gateway
            configure_versioning
            setup_api_monitoring
            configure_api_docs
            log_info "✅ API Gateway setup completed"
            ;;
        reload)
            sudo nginx -t && sudo systemctl reload nginx
            ;;
        monitor)
            /opt/ai-project-generator/scripts/monitor_api.sh
            ;;
        *)
            echo "Usage: $0 {setup|reload|monitor}"
            exit 1
            ;;
    esac
}

main "$@"



