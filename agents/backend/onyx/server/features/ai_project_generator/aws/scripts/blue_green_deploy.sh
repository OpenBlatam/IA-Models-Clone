#!/bin/bash
# Blue-Green Deployment Script
# Deploys new version alongside existing, then switches traffic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APP_DIR="/opt/ai-project-generator"
BLUE_DIR="/opt/ai-project-generator-blue"
GREEN_DIR="/opt/ai-project-generator-green"
CURRENT_COLOR_FILE="/opt/.deployment-color"

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

# Get current deployment color
get_current_color() {
    if [ -f "$CURRENT_COLOR_FILE" ]; then
        cat "$CURRENT_COLOR_FILE"
    else
        echo "blue"
    fi
}

# Get inactive color
get_inactive_color() {
    local current=$(get_current_color)
    if [ "$current" == "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Deploy to inactive environment
deploy_to_inactive() {
    local inactive_color=$(get_inactive_color)
    local inactive_dir="/opt/ai-project-generator-$inactive_color"
    
    log_info "Deploying to $inactive_color environment..."
    
    # Create directory
    sudo mkdir -p "$inactive_dir"
    sudo chown -R ubuntu:ubuntu "$inactive_dir"
    
    # Copy application files
    log_info "Copying application files..."
    sudo cp -r "$PROJECT_ROOT"/* "$inactive_dir/" || {
        log_error "Failed to copy files"
        return 1
    }
    
    # Install dependencies
    log_info "Installing dependencies..."
    cd "$inactive_dir"
    
    if [ -f "requirements.txt" ]; then
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    fi
    
    # Build Docker images if needed
    if [ -f "docker-compose.yml" ]; then
        log_info "Building Docker images..."
        sudo docker-compose -f "$inactive_dir/docker-compose.yml" build
    fi
    
    # Start services
    log_info "Starting $inactive_color environment..."
    if [ -f "docker-compose.yml" ]; then
        cd "$inactive_dir"
        sudo docker-compose up -d
    else
        sudo systemctl start "ai-project-generator-$inactive_color" || true
    fi
    
    # Health check
    log_info "Performing health check on $inactive_color environment..."
    local port=$([ "$inactive_color" == "blue" ] && echo "8020" || echo "8021")
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
            log_info "✅ $inactive_color environment is healthy"
            return 0
        else
            log_warn "Health check attempt $attempt/$max_attempts failed, retrying..."
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    log_error "❌ $inactive_color environment health check failed"
    return 1
}

# Switch traffic to new environment
switch_traffic() {
    local new_color=$(get_inactive_color)
    local old_color=$(get_current_color)
    
    log_info "Switching traffic from $old_color to $new_color..."
    
    # Update Nginx configuration
    if [ -f "/etc/nginx/sites-available/ai-project-generator" ]; then
        log_info "Updating Nginx configuration..."
        
        # Backup current config
        sudo cp /etc/nginx/sites-available/ai-project-generator \
               /etc/nginx/sites-available/ai-project-generator.backup
        
        # Update upstream to point to new color
        local new_port=$([ "$new_color" == "blue" ] && echo "8020" || echo "8021")
        sudo sed -i "s|proxy_pass http://.*:.*;|proxy_pass http://localhost:$new_port;|" \
            /etc/nginx/sites-available/ai-project-generator
        
        # Test and reload Nginx
        if sudo nginx -t; then
            sudo systemctl reload nginx
            log_info "✅ Nginx configuration updated"
        else
            log_error "❌ Nginx configuration test failed, reverting..."
            sudo cp /etc/nginx/sites-available/ai-project-generator.backup \
                   /etc/nginx/sites-available/ai-project-generator
            return 1
        fi
    fi
    
    # Update current color
    echo "$new_color" | sudo tee "$CURRENT_COLOR_FILE" > /dev/null
    
    log_info "✅ Traffic switched to $new_color environment"
    
    # Wait and verify
    sleep 10
    if curl -f -s "http://localhost/health" > /dev/null 2>&1; then
        log_info "✅ Health check passed after traffic switch"
        return 0
    else
        log_error "❌ Health check failed after traffic switch"
        return 1
    fi
}

# Cleanup old environment
cleanup_old_environment() {
    local old_color=$(get_current_color)
    local old_dir="/opt/ai-project-generator-$old_color"
    
    log_info "Cleaning up old $old_color environment..."
    
    # Stop services
    if [ -f "$old_dir/docker-compose.yml" ]; then
        cd "$old_dir"
        sudo docker-compose down || true
    else
        sudo systemctl stop "ai-project-generator-$old_color" || true
    fi
    
    # Keep directory for quick rollback (optional)
    # sudo rm -rf "$old_dir"
    
    log_info "✅ Old environment cleaned up"
}

# Rollback to previous environment
rollback() {
    local current_color=$(get_current_color)
    local previous_color=$(get_inactive_color)
    
    log_warn "Rolling back from $current_color to $previous_color..."
    
    # Switch traffic back
    if switch_traffic; then
        log_info "✅ Rollback successful"
        return 0
    else
        log_error "❌ Rollback failed"
        return 1
    fi
}

# Main function
main() {
    log_info "=== Blue-Green Deployment ==="
    
    case "${1:-deploy}" in
        deploy)
            if deploy_to_inactive && switch_traffic; then
                log_info "✅ Blue-green deployment successful"
                # Optionally cleanup old environment after verification period
                # cleanup_old_environment
            else
                log_error "❌ Blue-green deployment failed"
                exit 1
            fi
            ;;
        rollback)
            rollback
            ;;
        status)
            echo "Current deployment: $(get_current_color)"
            echo "Inactive deployment: $(get_inactive_color)"
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status}"
            exit 1
            ;;
    esac
}

main "$@"

