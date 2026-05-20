#!/bin/bash
# Canary Deployment Script
# Gradually rolls out new version to a percentage of traffic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APP_DIR="/opt/ai-project-generator"
CANARY_DIR="/opt/ai-project-generator-canary"
NGINX_CONFIG="/etc/nginx/sites-available/ai-project-generator"
CANARY_PERCENTAGE="${1:-10}"  # Default 10% traffic
MONITORING_DURATION="${2:-300}"  # Default 5 minutes

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

# Validate percentage
validate_percentage() {
    if ! [[ "$CANARY_PERCENTAGE" =~ ^[0-9]+$ ]] || [ "$CANARY_PERCENTAGE" -lt 1 ] || [ "$CANARY_PERCENTAGE" -gt 100 ]; then
        log_error "Canary percentage must be between 1 and 100"
        exit 1
    fi
}

# Deploy canary version
deploy_canary() {
    log_info "Deploying canary version ($CANARY_PERCENTAGE% traffic)..."
    
    # Create canary directory
    sudo mkdir -p "$CANARY_DIR"
    sudo chown -R ubuntu:ubuntu "$CANARY_DIR"
    
    # Copy application files
    log_info "Copying application files to canary..."
    sudo cp -r "$PROJECT_ROOT"/* "$CANARY_DIR/" || {
        log_error "Failed to copy files"
        return 1
    }
    
    # Install dependencies
    log_info "Installing dependencies..."
    cd "$CANARY_DIR"
    
    if [ -f "requirements.txt" ]; then
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    fi
    
    # Start canary services on different port
    local canary_port=8021
    
    if [ -f "docker-compose.yml" ]; then
        log_info "Starting canary Docker containers..."
        # Modify docker-compose to use different port
        sudo sed -i "s/8020:8020/$canary_port:$canary_port/g" "$CANARY_DIR/docker-compose.yml"
        cd "$CANARY_DIR"
        sudo docker-compose up -d
    else
        log_info "Starting canary service on port $canary_port..."
        # Update systemd service for canary
        sudo systemctl start "ai-project-generator-canary" || true
    fi
    
    # Health check
    log_info "Performing health check on canary..."
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$canary_port/health" > /dev/null 2>&1; then
            log_info "✅ Canary is healthy"
            return 0
        else
            log_warn "Health check attempt $attempt/$max_attempts failed, retrying..."
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    log_error "❌ Canary health check failed"
    return 1
}

# Update Nginx for canary traffic splitting
update_nginx_canary() {
    log_info "Updating Nginx for canary traffic splitting ($CANARY_PERCENTAGE%)..."
    
    # Backup current config
    sudo cp "$NGINX_CONFIG" "${NGINX_CONFIG}.backup.$(date +%s)"
    
    # Create canary configuration
    local canary_port=8021
    local main_weight=$((100 - CANARY_PERCENTAGE))
    local canary_weight=$CANARY_PERCENTAGE
    
    sudo tee "$NGINX_CONFIG" > /dev/null <<EOF
upstream app_backend {
    least_conn;
    server localhost:8020 weight=$main_weight;
    server localhost:$canary_port weight=$canary_weight;
}

server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://app_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Canary-Version "canary";
    }

    location /health {
        proxy_pass http://app_backend;
        access_log off;
    }
}
EOF
    
    # Test and reload Nginx
    if sudo nginx -t; then
        sudo systemctl reload nginx
        log_info "✅ Nginx updated for canary deployment"
    else
        log_error "❌ Nginx configuration test failed, reverting..."
        sudo cp "${NGINX_CONFIG}.backup.$(date +%s)" "$NGINX_CONFIG"
        return 1
    fi
}

# Monitor canary deployment
monitor_canary() {
    log_info "Monitoring canary deployment for $MONITORING_DURATION seconds..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + MONITORING_DURATION))
    local error_count=0
    local request_count=0
    
    while [ $(date +%s) -lt $end_time ]; do
        # Check error rate
        local errors=$(curl -s "http://localhost/health" 2>&1 | grep -c "error" || echo "0")
        error_count=$((error_count + errors))
        request_count=$((request_count + 1))
        
        # Check response time
        local response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://localhost/health" 2>/dev/null || echo "0")
        
        if (( $(echo "$response_time > 2.0" | bc -l) )); then
            log_warn "High response time detected: ${response_time}s"
        fi
        
        sleep 10
    done
    
    local error_rate=$((error_count * 100 / request_count))
    
    if [ $error_rate -gt 5 ]; then
        log_error "❌ High error rate detected: ${error_rate}%"
        return 1
    else
        log_info "✅ Canary monitoring passed (error rate: ${error_rate}%)"
        return 0
    fi
}

# Promote canary to production
promote_canary() {
    log_info "Promoting canary to production..."
    
    # Update Nginx to route 100% to canary
    sudo sed -i 's/weight=[0-9]*/weight=100/g' "$NGINX_CONFIG"
    sudo sed -i 's/localhost:8020 weight=[0-9]*/localhost:8021 weight=0/g' "$NGINX_CONFIG"
    
    if sudo nginx -t; then
        sudo systemctl reload nginx
        log_info "✅ Canary promoted to production"
        
        # Stop old version
        if [ -f "$APP_DIR/docker-compose.yml" ]; then
            cd "$APP_DIR"
            sudo docker-compose down || true
        else
            sudo systemctl stop "ai-project-generator" || true
        fi
        
        # Move canary to production
        sudo mv "$CANARY_DIR" "$APP_DIR.new"
        sudo systemctl restart "ai-project-generator" || true
        
        return 0
    else
        log_error "❌ Failed to promote canary"
        return 1
    fi
}

# Rollback canary
rollback_canary() {
    log_warn "Rolling back canary deployment..."
    
    # Restore original Nginx config
    local latest_backup=$(ls -t "${NGINX_CONFIG}.backup."* 2>/dev/null | head -1)
    if [ -n "$latest_backup" ]; then
        sudo cp "$latest_backup" "$NGINX_CONFIG"
        sudo nginx -t && sudo systemctl reload nginx
    fi
    
    # Stop canary
    if [ -f "$CANARY_DIR/docker-compose.yml" ]; then
        cd "$CANARY_DIR"
        sudo docker-compose down || true
    else
        sudo systemctl stop "ai-project-generator-canary" || true
    fi
    
    log_info "✅ Canary rolled back"
}

# Main function
main() {
    log_info "=== Canary Deployment ==="
    log_info "Canary percentage: $CANARY_PERCENTAGE%"
    log_info "Monitoring duration: $MONITORING_DURATION seconds"
    
    validate_percentage
    
    case "${3:-deploy}" in
        deploy)
            if deploy_canary && update_nginx_canary; then
                log_info "✅ Canary deployed successfully"
                log_info "Monitoring canary for $MONITORING_DURATION seconds..."
                
                if monitor_canary; then
                    read -p "Promote canary to production? (y/N) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        promote_canary
                    else
                        log_info "Canary deployment kept at $CANARY_PERCENTAGE%"
                    fi
                else
                    log_error "Canary monitoring failed, rolling back..."
                    rollback_canary
                    exit 1
                fi
            else
                log_error "Canary deployment failed"
                exit 1
            fi
            ;;
        promote)
            promote_canary
            ;;
        rollback)
            rollback_canary
            ;;
        monitor)
            monitor_canary
            ;;
        *)
            echo "Usage: $0 <percentage> <monitoring_duration> {deploy|promote|rollback|monitor}"
            exit 1
            ;;
    esac
}

main "$@"

