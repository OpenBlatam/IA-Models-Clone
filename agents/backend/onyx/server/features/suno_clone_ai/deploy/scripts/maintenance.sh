#!/bin/bash
# Maintenance script for Suno Clone AI
# Performs routine maintenance tasks

set -euo pipefail

# Configuration
readonly CONTAINER_NAME="suno-clone-ai"
readonly PROJECT_DIR="${HOME}/suno-clone-ai"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Clean Docker system
clean_docker() {
    log_info "Cleaning Docker system..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused containers
    docker container prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    log_info "Docker cleanup completed"
}

# Clean logs
clean_logs() {
    log_info "Cleaning old logs..."
    
    # Clean container logs (keep last 100MB)
    if [ -f "/var/lib/docker/containers/$(docker inspect -f '{{.Id}}' ${CONTAINER_NAME})/$(docker inspect -f '{{.Id}}' ${CONTAINER_NAME})-json.log" ]; then
        truncate -s 100M "/var/lib/docker/containers/$(docker inspect -f '{{.Id}}' ${CONTAINER_NAME})/$(docker inspect -f '{{.Id}}' ${CONTAINER_NAME})-json.log" 2>/dev/null || true
    fi
    
    # Clean application logs
    find "${PROJECT_DIR}/logs" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    log_info "Log cleanup completed"
}

# Optimize database
optimize_database() {
    log_info "Optimizing database..."
    
    docker exec "${CONTAINER_NAME}" python -c "
import sqlite3
conn = sqlite3.connect('/app/suno_clone.db')
conn.execute('VACUUM')
conn.execute('ANALYZE')
conn.close()
" 2>/dev/null || {
        log_warn "Database optimization skipped (database may not exist)"
    }
    
    log_info "Database optimization completed"
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    
    apt-get update
    apt-get upgrade -y
    
    log_info "System packages updated"
}

# Check disk space
check_disk_space() {
    log_info "Checking disk space..."
    
    df -h | grep -E '^/dev/' | while read line; do
        local usage=$(echo $line | awk '{print $5}' | sed 's/%//')
        local mount=$(echo $line | awk '{print $6}')
        
        if [ "${usage}" -gt 80 ]; then
            log_warn "High disk usage on ${mount}: ${usage}%"
        else
            log_info "Disk usage on ${mount}: ${usage}%"
        fi
    done
}

# Run all maintenance tasks
run_all() {
    log_info "=== Starting Maintenance ==="
    
    clean_docker
    clean_logs
    optimize_database
    check_disk_space
    
    log_info "=== Maintenance Completed ==="
}

# Main function
main() {
    case "${1:-all}" in
        docker)
            clean_docker
            ;;
        logs)
            clean_logs
            ;;
        database)
            optimize_database
            ;;
        system)
            update_system
            ;;
        disk)
            check_disk_space
            ;;
        all|*)
            run_all
            ;;
    esac
}

main "$@"




