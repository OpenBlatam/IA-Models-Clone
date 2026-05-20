#!/bin/bash
# Update script for EC2 deployment
# Safely updates the application with zero-downtime
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
readonly BACKUP_DIR="/opt/music-analyzer-ai-backup"
readonly HEALTH_URL="http://localhost:8010/health"

log_step "Music Analyzer AI - Update Script"

# Create backup
create_backup() {
    log_step "Creating Backup"
    
    if [ ! -d "$APP_DIR" ]; then
        error_exit "Application directory not found: $APP_DIR"
    fi
    
    if [ -d "$BACKUP_DIR" ]; then
        log_info "Removing old backup..."
        sudo rm -rf "$BACKUP_DIR"
    fi
    
    log_info "Creating backup at $BACKUP_DIR..."
    sudo cp -r "$APP_DIR" "$BACKUP_DIR" || error_exit "Failed to create backup"
    log_success "Backup created successfully"
}

# Pull latest code
pull_latest_code() {
    log_step "Pulling Latest Code"
    
    cd "$APP_DIR" || error_exit "Cannot change to directory: $APP_DIR"
    
    if [ -d ".git" ]; then
        log_info "Pulling latest code from Git..."
        if git pull origin main 2>/dev/null || git pull origin master 2>/dev/null; then
            log_success "Code updated from Git"
        else
            log_warning "Git pull failed, continuing with existing code..."
        fi
    else
        log_info "Not a Git repository, skipping pull"
    fi
}

# Perform rolling update
rolling_update() {
    log_step "Performing Rolling Update"
    
    cd "$APP_DIR" || error_exit "Cannot change to directory: $APP_DIR"
    
    local compose_cmd=$(docker_compose_cmd) || error_exit "Docker Compose not available"
    
    # Services to update in order
    local services=("redis" "postgres" "music-analyzer-ai")
    
    for service in "${services[@]}"; do
        if $compose_cmd -f deployment/docker-compose.prod.yml ps | grep -q "$service"; then
            log_info "Updating $service..."
            $compose_cmd -f deployment/docker-compose.prod.yml up -d --no-deps "$service" || {
                log_warning "Failed to update $service, continuing..."
            }
            sleep 5
        fi
    done
    
    # Restart all services
    log_info "Restarting all services..."
    $compose_cmd -f deployment/docker-compose.prod.yml up -d
    log_success "Rolling update complete"
}

# Rollback to backup
rollback() {
    log_step "Rolling Back to Previous Version"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        error_exit "Backup directory not found: $BACKUP_DIR"
    fi
    
    log_info "Stopping current services..."
    stop_services "$APP_DIR"
    
    log_info "Restoring from backup..."
    sudo rm -rf "$APP_DIR"
    sudo mv "$BACKUP_DIR" "$APP_DIR"
    
    log_info "Starting restored services..."
    start_services "$APP_DIR"
    
    log_success "Rollback complete"
}

# Main update function
main() {
    # Pre-flight checks
    ensure_docker
    ensure_docker_compose
    
    # Create backup
    create_backup
    
    # Pull latest code
    pull_latest_code
    
    # Rebuild images
    log_step "Rebuilding Docker Images"
    build_images "$APP_DIR" 1800
    
    # Perform rolling update
    rolling_update
    
    # Wait for health check
    if wait_for_service "$HEALTH_URL" 30 2; then
        log_success "Update successful!"
        
        # Cleanup
        log_info "Cleaning up old Docker images..."
        docker_cleanup
        
        log_success "Update complete!"
    else
        log_error "Health check failed. Rolling back..."
        rollback
        error_exit "Update failed. Rolled back to previous version."
    fi
}

# Run main function
main "$@"

