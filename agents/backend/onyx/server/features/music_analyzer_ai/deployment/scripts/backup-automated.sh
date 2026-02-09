#!/bin/bash
# Automated Backup Script
# Comprehensive backup with rotation and verification

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/cloud.sh"

# Initialize
init_common

# Configuration
readonly BACKUP_DIR="${BACKUP_DIR:-/backups/music-analyzer-ai}"
readonly RETENTION_DAYS="${RETENTION_DAYS:-30}"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly BACKUP_NAME="backup-${TIMESTAMP}"
readonly BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
readonly VERIFY_BACKUP="${VERIFY_BACKUP:-true}"
readonly UPLOAD_TO_CLOUD="${UPLOAD_TO_CLOUD:-false}"

# Create backup directory
mkdir -p "${BACKUP_PATH}"

# Backup database
backup_database() {
    log_info "Backing up database..."
    
    local db_type="${DATABASE_TYPE:-postgresql}"
    local db_url="${DATABASE_URL:-}"
    
    if [ -z "${db_url}" ]; then
        log_warn "DATABASE_URL not set, skipping database backup"
        return 0
    fi
    
    case "${db_type}" in
        postgresql)
            if command -v pg_dump &> /dev/null; then
                pg_dump "${db_url}" > "${BACKUP_PATH}/database.sql" || {
                    log_error "Database backup failed"
                    return 1
                }
                log_success "Database backup completed"
            else
                log_warn "pg_dump not available, skipping database backup"
            fi
            ;;
        mysql)
            if command -v mysqldump &> /dev/null; then
                mysqldump "${db_url}" > "${BACKUP_PATH}/database.sql" || {
                    log_error "Database backup failed"
                    return 1
                }
                log_success "Database backup completed"
            else
                log_warn "mysqldump not available, skipping database backup"
            fi
            ;;
        *)
            log_warn "Unknown database type: ${db_type}"
            ;;
    esac
}

# Backup Redis
backup_redis() {
    log_info "Backing up Redis..."
    
    local redis_url="${REDIS_URL:-redis://localhost:6379}"
    
    if command -v redis-cli &> /dev/null; then
        redis-cli --rdb "${BACKUP_PATH}/redis.rdb" -u "${redis_url}" || {
            log_warn "Redis backup failed or Redis not available"
        }
        log_success "Redis backup completed"
    else
        log_warn "redis-cli not available, skipping Redis backup"
    fi
}

# Backup configuration files
backup_config() {
    log_info "Backing up configuration files..."
    
    local config_dir="${BACKUP_PATH}/config"
    mkdir -p "${config_dir}"
    
    # Backup .env files
    find . -name ".env*" -type f -exec cp {} "${config_dir}/" \; 2>/dev/null || true
    
    # Backup Kubernetes configs
    if [ -d "deployment/kubernetes" ]; then
        cp -r deployment/kubernetes "${config_dir}/" || true
    fi
    
    # Backup Ansible configs
    if [ -d "deployment/ansible" ]; then
        cp -r deployment/ansible "${config_dir}/" || true
    fi
    
    log_success "Configuration backup completed"
}

# Backup Docker volumes
backup_volumes() {
    log_info "Backing up Docker volumes..."
    
    local volumes=("music-analyzer-ai-data" "music-analyzer-ai-storage")
    
    for volume in "${volumes[@]}"; do
        if docker volume inspect "${volume}" &> /dev/null; then
            log_info "Backing up volume: ${volume}"
            docker run --rm \
                -v "${volume}:/data:ro" \
                -v "${BACKUP_PATH}:/backup" \
                alpine tar czf "/backup/${volume}.tar.gz" -C /data . || {
                log_warn "Failed to backup volume: ${volume}"
            }
        fi
    done
    
    log_success "Volume backup completed"
}

# Verify backup
verify_backup() {
    if [ "${VERIFY_BACKUP}" != "true" ]; then
        return 0
    fi
    
    log_info "Verifying backup..."
    
    local issues=0
    
    # Check database backup
    if [ -f "${BACKUP_PATH}/database.sql" ]; then
        if [ ! -s "${BACKUP_PATH}/database.sql" ]; then
            log_error "Database backup is empty"
            issues=$((issues + 1))
        fi
    fi
    
    # Check Redis backup
    if [ -f "${BACKUP_PATH}/redis.rdb" ]; then
        if [ ! -s "${BACKUP_PATH}/redis.rdb" ]; then
            log_error "Redis backup is empty"
            issues=$((issues + 1))
        fi
    fi
    
    if [ ${issues} -eq 0 ]; then
        log_success "Backup verification passed"
        return 0
    else
        log_error "Backup verification failed"
        return 1
    fi
}

# Compress backup
compress_backup() {
    log_info "Compressing backup..."
    
    cd "${BACKUP_DIR}"
    tar czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}" || {
        log_error "Failed to compress backup"
        return 1
    }
    
    rm -rf "${BACKUP_NAME}"
    
    log_success "Backup compressed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# Upload to cloud
upload_to_cloud() {
    if [ "${UPLOAD_TO_CLOUD}" != "true" ]; then
        return 0
    fi
    
    log_info "Uploading backup to cloud..."
    
    local backup_file="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    
    # Upload to S3
    if [ -n "${S3_BACKUP_BUCKET:-}" ] && aws_check_credentials; then
        aws_upload_to_s3 "${backup_file}" \
            "${S3_BACKUP_BUCKET}/music-analyzer-ai/${BACKUP_NAME}.tar.gz" || {
            log_warn "Failed to upload to S3"
        }
    fi
    
    # Upload to Azure Storage
    if [ -n "${AZURE_STORAGE_ACCOUNT:-}" ] && azure_check_credentials; then
        azure_upload_to_storage "${backup_file}" \
            "${AZURE_STORAGE_ACCOUNT}" \
            "backups" \
            "music-analyzer-ai/${BACKUP_NAME}.tar.gz" || {
            log_warn "Failed to upload to Azure Storage"
        }
    fi
    
    log_success "Cloud upload completed"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than ${RETENTION_DAYS} days..."
    
    find "${BACKUP_DIR}" -name "backup-*.tar.gz" \
        -type f \
        -mtime +${RETENTION_DAYS} \
        -delete || true
    
    log_success "Old backups cleaned up"
}

# Main function
main() {
    log_info "Starting automated backup..."
    log_info "Backup directory: ${BACKUP_DIR}"
    log_info "Retention: ${RETENTION_DAYS} days"
    
    backup_database
    backup_redis
    backup_config
    backup_volumes
    verify_backup
    compress_backup
    upload_to_cloud
    cleanup_old_backups
    
    log_success "Automated backup completed!"
    log_info "Backup location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

main "$@"




