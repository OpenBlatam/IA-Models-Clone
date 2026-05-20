#!/bin/bash
# Rollback Script - Restores previous deployment version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BACKUP_DIR="/opt/backups"
APP_DIR="/opt/ai-project-generator"
MAX_BACKUPS=10

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# List available backups
list_backups() {
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    log_info "Available backups:"
    ls -lth "$BACKUP_DIR" | grep -E "backup-.*\.tar\.gz" | head -$MAX_BACKUPS | nl
}

# Get latest backup
get_latest_backup() {
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory not found"
        return 1
    fi
    
    local latest=$(ls -t "$BACKUP_DIR"/backup-*.tar.gz 2>/dev/null | head -1)
    
    if [ -z "$latest" ]; then
        log_error "No backups found"
        return 1
    fi
    
    echo "$latest"
}

# Restore from backup
restore_backup() {
    local backup_file="${1:-}"
    
    if [ -z "$backup_file" ]; then
        backup_file=$(get_latest_backup)
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "Restoring from backup: $backup_file"
    
    # Stop application
    log_info "Stopping application..."
    sudo systemctl stop ai-project-generator || true
    sudo docker-compose -f "$APP_DIR/docker-compose.yml" down || true
    
    # Backup current state before rollback
    local current_backup="$BACKUP_DIR/pre-rollback-$(date +%Y%m%d-%H%M%S).tar.gz"
    log_info "Creating pre-rollback backup: $current_backup"
    sudo tar -czf "$current_backup" -C "$APP_DIR" . 2>/dev/null || true
    
    # Restore from backup
    log_info "Extracting backup..."
    sudo mkdir -p "$APP_DIR"
    sudo tar -xzf "$backup_file" -C "$APP_DIR"
    
    # Restore permissions
    sudo chown -R ubuntu:ubuntu "$APP_DIR"
    
    # Restart application
    log_info "Restarting application..."
    if [ -f "$APP_DIR/docker-compose.yml" ]; then
        cd "$APP_DIR"
        sudo docker-compose up -d
    else
        sudo systemctl start ai-project-generator
    fi
    
    # Wait for application to start
    log_info "Waiting for application to start..."
    sleep 10
    
    # Health check
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:8020/health" > /dev/null 2>&1; then
            log_info "✅ Rollback successful! Application is healthy."
            return 0
        else
            log_warn "Health check attempt $attempt/$max_attempts failed, retrying..."
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    log_error "❌ Rollback completed but health check failed"
    return 1
}

# Main function
main() {
    log_info "=== Rollback Script ==="
    
    if [ "${1:-}" == "--list" ]; then
        list_backups
        exit 0
    fi
    
    local backup_file="${1:-}"
    
    if [ -n "$backup_file" ] && [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        log_info "Use --list to see available backups"
        exit 1
    fi
    
    if ! restore_backup "$backup_file"; then
        log_error "Rollback failed"
        exit 1
    fi
    
    log_info "=== Rollback completed ==="
}

main "$@"

