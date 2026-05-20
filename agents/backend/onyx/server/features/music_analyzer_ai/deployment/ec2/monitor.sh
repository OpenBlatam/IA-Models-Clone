#!/bin/bash
# Monitoring and health check script for EC2 deployment
# Refactored with modular functions

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

# Source libraries
source "$LIB_DIR/common.sh" || error_exit "Failed to load common.sh"
source "$LIB_DIR/docker.sh" || error_exit "Failed to load docker.sh"
source "$LIB_DIR/deployment.sh" || error_exit "Failed to load deployment.sh"

# Configuration
readonly APP_DIR="/opt/music-analyzer-ai"
readonly HEALTH_URL="http://localhost:8010/health"

log_step "Music Analyzer AI - System Monitor"

# Check Docker status
check_docker_status() {
    log_info "Docker Status:"
    if docker_is_running; then
        log_success "Docker is running"
        echo "  Version: $(docker --version)"
        echo "  Containers: $(docker ps -q | wc -l) running"
    else
        log_error "Docker is not running"
    fi
    echo ""
}

# Check container status
check_container_status() {
    log_info "Container Status:"
    if [ ! -d "$APP_DIR" ]; then
        log_error "Application directory not found: $APP_DIR"
        return 1
    fi
    
    local status=$(get_service_status "$APP_DIR")
    if echo "$status" | grep -q "Up"; then
        log_success "Containers are running:"
        echo "$status"
    else
        log_warning "Some containers may not be running:"
        echo "$status"
    fi
    echo ""
}

# Check health endpoint
check_health_endpoint() {
    log_info "Health Check:"
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        log_success "Health endpoint is responding"
        local health_response=$(curl -s "$HEALTH_URL")
        if command_exists python3 && echo "$health_response" | python3 -m json.tool >/dev/null 2>&1; then
            echo "$health_response" | python3 -m json.tool
        else
            echo "$health_response"
        fi
    else
        log_error "Health endpoint is not responding"
    fi
    echo ""
}

# Check system resources
check_system_resources() {
    log_info "System Resources:"
    
    if command_exists top; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' || echo "N/A")
        echo "  CPU Usage: $cpu_usage"
    fi
    
    if command_exists free; then
        local memory=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
        echo "  Memory: $memory"
    fi
    
    if command_exists df; then
        local disk=$(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')
        echo "  Disk: $disk"
    fi
    echo ""
}

# Check Docker resources
check_docker_resources() {
    log_info "Docker Resources:"
    if docker_stats; then
        echo ""
    else
        log_warning "Could not get Docker stats"
    fi
}

# Check port status
check_port_status() {
    log_info "Port Status:"
    local ports=(8010 80 443 3000 9090)
    
    for port in "${ports[@]}"; do
        if (netstat -tuln 2>/dev/null | grep -q ":$port ") || (ss -tuln 2>/dev/null | grep -q ":$port "); then
            log_success "Port $port is listening"
        else
            log_warning "Port $port is not listening"
        fi
    done
    echo ""
}

# Show recent logs
show_recent_logs() {
    log_info "Recent Logs (last 20 lines):"
    get_service_logs "$APP_DIR" "" 20 || log_warning "Could not retrieve logs"
    echo ""
}

# Main monitoring function
main() {
    check_docker_status
    check_container_status
    check_health_endpoint
    check_system_resources
    check_docker_resources
    check_port_status
    show_recent_logs
    
    log_success "Monitoring complete!"
}

# Run main function
main "$@"

