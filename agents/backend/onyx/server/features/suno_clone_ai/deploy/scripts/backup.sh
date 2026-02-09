#!/bin/bash
# Backup script for Suno Clone AI
# Creates backups of database, audio files, and configuration

set -euo pipefail

# Configuration
readonly BACKUP_DIR="${BACKUP_DIR:-/backups/suno-clone-ai}"
readonly RETENTION_DAYS="${RETENTION_DAYS:-7}"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly BACKUP_NAME="suno-clone-ai-backup-${TIMESTAMP}"
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

# Create backup directory
create_backup_dir() {
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
    log_info "Created backup directory: ${BACKUP_DIR}/${BACKUP_NAME}"
}

# Backup database
backup_database() {
    log_info "Backing up database..."
    
    if docker exec "${CONTAINER_NAME}" test -f /app/suno_clone.db; then
        docker cp "${CONTAINER_NAME}:/app/suno_clone.db" \
            "${BACKUP_DIR}/${BACKUP_NAME}/suno_clone.db"
        log_info "Database backup completed"
    else
        log_warn "Database file not found, skipping database backup"
    fi
}

# Backup audio files
backup_audio() {
    log_info "Backing up audio files..."
    
    if docker exec "${CONTAINER_NAME}" test -d /app/storage/audio; then
        docker cp "${CONTAINER_NAME}:/app/storage/audio" \
            "${BACKUP_DIR}/${BACKUP_NAME}/audio"
        log_info "Audio files backup completed"
    else
        log_warn "Audio directory not found, skipping audio backup"
    fi
}

# Backup configuration
backup_config() {
    log_info "Backing up configuration..."
    
    local config_dir="${BACKUP_DIR}/${BACKUP_NAME}/config"
    mkdir -p "${config_dir}"
    
    # Backup .env file if exists
    if [ -f "${HOME}/suno-clone-ai/.env" ]; then
        cp "${HOME}/suno-clone-ai/.env" "${config_dir}/.env"
        log_info "Configuration backup completed"
    else
        log_warn ".env file not found, skipping configuration backup"
    fi
}

# Backup Docker volumes (if any)
backup_volumes() {
    log_info "Backing up Docker volumes..."
    
    docker run --rm \
        -v suno-clone-ai-storage:/data:ro \
        -v "${BACKUP_DIR}/${BACKUP_NAME}:/backup" \
        alpine tar czf /backup/volumes.tar.gz -C /data . 2>/dev/null || {
        log_warn "No Docker volumes found, skipping volume backup"
    }
}

# Compress backup
compress_backup() {
    log_info "Compressing backup..."
    
    cd "${BACKUP_DIR}"
    tar czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
    rm -rf "${BACKUP_NAME}"
    
    log_info "Backup compressed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# Upload to S3 (optional)
upload_to_s3() {
    if [ -n "${S3_BACKUP_BUCKET:-}" ] && command -v aws &> /dev/null; then
        log_info "Uploading backup to S3..."
        
        aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
            "s3://${S3_BACKUP_BUCKET}/suno-clone-ai/${BACKUP_NAME}.tar.gz" || {
            log_warn "Failed to upload to S3, continuing..."
        }
        
        log_info "Backup uploaded to S3"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than ${RETENTION_DAYS} days..."
    
    find "${BACKUP_DIR}" -name "suno-clone-ai-backup-*.tar.gz" \
        -type f -mtime +${RETENTION_DAYS} -delete || true
    
    log_info "Old backups cleaned up"
}

# Main function
main() {
    log_info "Starting backup process..."
    
    create_backup_dir
    backup_database
    backup_audio
    backup_config
    backup_volumes
    compress_backup
    upload_to_s3
    cleanup_old_backups
    
    log_info "✅ Backup completed successfully!"
    log_info "Backup location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# Run main function
main "$@"




