#!/bin/bash
# Restore script for Suno Clone AI
# Restores from backup files

set -euo pipefail

# Configuration
readonly BACKUP_DIR="${BACKUP_DIR:-/backups/suno-clone-ai}"
readonly CONTAINER_NAME="suno-clone-ai"

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

# List available backups
list_backups() {
    log_info "Available backups:"
    ls -lh "${BACKUP_DIR}"/*.tar.gz 2>/dev/null | awk '{print $9, $5}' || {
        log_error "No backups found in ${BACKUP_DIR}"
        exit 1
    }
}

# Restore database
restore_database() {
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    log_info "Extracting backup..."
    tar xzf "${backup_file}" -C "${temp_dir}"
    
    log_info "Restoring database..."
    if [ -f "${temp_dir}"/*/suno_clone.db ]; then
        docker cp "${temp_dir}"/*/suno_clone.db \
            "${CONTAINER_NAME}:/app/suno_clone.db"
        log_info "Database restored"
    else
        log_warn "Database file not found in backup"
    fi
    
    rm -rf "${temp_dir}"
}

# Restore audio files
restore_audio() {
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    log_info "Extracting backup..."
    tar xzf "${backup_file}" -C "${temp_dir}"
    
    log_info "Restoring audio files..."
    if [ -d "${temp_dir}"/*/audio ]; then
        docker cp "${temp_dir}"/*/audio \
            "${CONTAINER_NAME}:/app/storage/"
        log_info "Audio files restored"
    else
        log_warn "Audio directory not found in backup"
    fi
    
    rm -rf "${temp_dir}"
}

# Restore configuration
restore_config() {
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    log_info "Extracting backup..."
    tar xzf "${backup_file}" -C "${temp_dir}"
    
    log_info "Restoring configuration..."
    if [ -f "${temp_dir}"/*/config/.env ]; then
        cp "${temp_dir}"/*/config/.env "${HOME}/suno-clone-ai/.env"
        log_info "Configuration restored"
        log_warn "Please restart the container for configuration changes to take effect"
    else
        log_warn "Configuration file not found in backup"
    fi
    
    rm -rf "${temp_dir}"
}

# Main restore function
main() {
    if [ $# -eq 0 ]; then
        list_backups
        echo ""
        echo "Usage: $0 <backup-file.tar.gz> [--database] [--audio] [--config]"
        echo "Example: $0 suno-clone-ai-backup-20240101_120000.tar.gz"
        exit 1
    fi
    
    local backup_file="$1"
    shift
    
    if [ ! -f "${backup_file}" ]; then
        # Try in backup directory
        backup_file="${BACKUP_DIR}/${backup_file}"
        if [ ! -f "${backup_file}" ]; then
            log_error "Backup file not found: ${backup_file}"
            exit 1
        fi
    fi
    
    log_info "Restoring from: ${backup_file}"
    
    # If no specific options, restore all
    if [ $# -eq 0 ]; then
        restore_database "${backup_file}"
        restore_audio "${backup_file}"
        restore_config "${backup_file}"
    else
        # Restore specific components
        while [ $# -gt 0 ]; do
            case "$1" in
                --database)
                    restore_database "${backup_file}"
                    ;;
                --audio)
                    restore_audio "${backup_file}"
                    ;;
                --config)
                    restore_config "${backup_file}"
                    ;;
                *)
                    log_error "Unknown option: $1"
                    exit 1
                    ;;
            esac
            shift
        done
    fi
    
    log_info "✅ Restore completed!"
    log_warn "Please restart the container: docker restart ${CONTAINER_NAME}"
}

main "$@"




