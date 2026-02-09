#!/bin/bash

###############################################################################
# Automated Backup Script for AI Project Generator
# Performs automated backups of application data, logs, and configurations
# Improved version with better error handling, integrity checks, and metrics
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh" 2>/dev/null || {
    echo "Error: common_functions.sh not found" >&2
    exit 1
}

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/backups/ai-project-generator}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-backups/}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
APP_DIR="${APP_DIR:-/opt/ai-project-generator}"
LOG_FILE="${LOG_FILE:-/var/log/backup.log}"
CLOUDWATCH_NAMESPACE="${CLOUDWATCH_NAMESPACE:-AIProjectGenerator/Backups}"
ENABLE_COMPRESSION="${ENABLE_COMPRESSION:-true}"
VERIFY_INTEGRITY="${VERIFY_INTEGRITY:-true}"
MIN_DISK_SPACE_GB="${MIN_DISK_SPACE_GB:-5}"

# Date stamp
DATE_STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ai-project-generator-backup-${DATE_STAMP}"
BACKUP_START_TIME=$(date +%s)

# Metrics
BACKUP_SIZE=0
BACKUP_DURATION=0
BACKUP_SUCCESS=false

###############################################################################
# Cleanup and Error Handling
###############################################################################

cleanup() {
    local exit_code=$?
    local end_time=$(date +%s)
    BACKUP_DURATION=$((end_time - BACKUP_START_TIME))
    
    if [ $exit_code -ne 0 ]; then
        log_error "Backup failed with exit code: $exit_code"
        BACKUP_SUCCESS=false
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "BackupFailed" 1
    else
        BACKUP_SUCCESS=true
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "BackupSuccess" 1
    fi
    
    # Send duration metric
    send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "BackupDuration" \
        "${BACKUP_DURATION}" "Seconds"
    
    # Send size metric if available
    if [ $BACKUP_SIZE -gt 0 ]; then
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "BackupSize" \
            "${BACKUP_SIZE}" "Bytes"
    fi
    
    exit $exit_code
}

trap cleanup EXIT INT TERM

###############################################################################
# Validation
###############################################################################

validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check if backup directory exists
    if ! check_directory_exists "${BACKUP_DIR}"; then
        log_info "Creating backup directory: ${BACKUP_DIR}"
        mkdir -p "${BACKUP_DIR}" || {
            log_error "Failed to create backup directory"
            exit 1
        }
    fi
    
    # Check disk space
    if ! check_disk_space "${BACKUP_DIR}" "${MIN_DISK_SPACE_GB}"; then
        log_error "Insufficient disk space for backup"
        exit 1
    fi
    
    # Check if app directory exists
    if ! check_directory_exists "${APP_DIR}"; then
        log_error "Application directory not found: ${APP_DIR}"
        exit 1
    fi
    
    # Check AWS CLI if S3 backup is enabled
    if [ -n "${S3_BUCKET}" ]; then
        if ! check_command "aws"; then
            log_error "AWS CLI not found. Install it or disable S3 backup."
            exit 1
        fi
        
        if ! check_aws_credentials; then
            log_error "AWS credentials not configured"
            exit 1
        fi
        
        if ! aws s3 ls "s3://${S3_BUCKET}" &> /dev/null; then
            log_error "Cannot access S3 bucket: ${S3_BUCKET}"
            exit 1
        fi
    fi
    
    log_success "Prerequisites validated."
}

###############################################################################
# Backup Functions
###############################################################################

backup_application_data() {
    log_info "Backing up application data..."
    
    local backup_path="${BACKUP_DIR}/${BACKUP_NAME}"
    mkdir -p "${backup_path}" || {
        log_error "Failed to create backup directory"
        return 1
    }
    
    # Create backup marker
    create_backup_marker "${backup_path}" > /dev/null
    
    # Backup application directory (excluding large files)
    log_info "Backing up application files..."
    local tar_opts="-czf"
    if [ "${ENABLE_COMPRESSION}" != "true" ]; then
        tar_opts="-cf"
    fi
    
    if tar ${tar_opts} "${backup_path}/application.tar.gz" \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='*.log' \
        --exclude='.pytest_cache' \
        --exclude='.mypy_cache' \
        --exclude='venv' \
        --exclude='.venv' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        -C "$(dirname ${APP_DIR})" \
        "$(basename ${APP_DIR})" 2>/dev/null; then
        log_success "Application files backed up"
    else
        log_warn "Some files were excluded from backup or backup had warnings"
    fi
    
    # Backup environment file
    if check_file_exists "${APP_DIR}/.env"; then
        cp "${APP_DIR}/.env" "${backup_path}/.env" && \
            log_info "Environment file backed up"
    fi
    
    # Backup Docker Compose file
    if check_file_exists "${APP_DIR}/docker-compose.yml"; then
        cp "${APP_DIR}/docker-compose.yml" "${backup_path}/docker-compose.yml"
    fi
    
    # Backup database (if using local database)
    if check_command "pg_dump"; then
        log_info "Backing up PostgreSQL database..."
        if pg_dump -U postgres ai_project_generator > "${backup_path}/database.sql" 2>/dev/null; then
            log_success "PostgreSQL database backed up"
        else
            log_warn "Database backup skipped (database may not exist or connection failed)"
        fi
    fi
    
    # Backup Redis data
    if check_command "redis-cli"; then
        log_info "Backing up Redis data..."
        if redis-cli --rdb "${backup_path}/redis.rdb" 2>/dev/null; then
            log_success "Redis data backed up"
        else
            log_warn "Redis backup skipped (Redis may not be running)"
        fi
    fi
    
    # Verify backup integrity
    if [ "${VERIFY_INTEGRITY}" = "true" ]; then
        if verify_backup_integrity "${backup_path}"; then
            log_success "Backup integrity verified"
        else
            log_error "Backup integrity check failed"
            return 1
        fi
    fi
    
    # Calculate backup size
    BACKUP_SIZE=$(du -sb "${backup_path}" 2>/dev/null | cut -f1)
    local backup_size_human=$(calculate_directory_size "${backup_path}")
    
    log_success "Application data backup completed: ${backup_path} (${backup_size_human})"
    echo "${backup_path}" > "${BACKUP_DIR}/latest_backup.txt"
}

backup_logs() {
    log_info "Backing up logs..."
    
    local backup_path="${BACKUP_DIR}/${BACKUP_NAME}"
    mkdir -p "${backup_path}/logs"
    
    # Backup application logs
    if [ -d "${APP_DIR}/logs" ]; then
        tar -czf "${backup_path}/logs/application_logs.tar.gz" \
            -C "${APP_DIR}" logs/ 2>/dev/null || true
    fi
    
    # Backup system logs
    if command -v journalctl &> /dev/null; then
        journalctl -u ai-project-generator --since "7 days ago" \
            > "${backup_path}/logs/system.log" 2>/dev/null || true
    fi
    
    # Backup Nginx logs
    if [ -d "/var/log/nginx" ]; then
        tar -czf "${backup_path}/logs/nginx_logs.tar.gz" \
            /var/log/nginx/*.log 2>/dev/null || true
    fi
    
    log_info "Logs backup completed"
}

backup_configurations() {
    log_info "Backing up configurations..."
    
    local backup_path="${BACKUP_DIR}/${BACKUP_NAME}"
    mkdir -p "${backup_path}/config"
    
    # Backup Nginx configuration
    if [ -f "/etc/nginx/sites-available/ai-project-generator" ]; then
        cp "/etc/nginx/sites-available/ai-project-generator" \
            "${backup_path}/config/nginx.conf"
    fi
    
    # Backup systemd service file
    if [ -f "/etc/systemd/system/ai-project-generator.service" ]; then
        cp "/etc/systemd/system/ai-project-generator.service" \
            "${backup_path}/config/systemd.service"
    fi
    
    # Backup Redis configuration
    if [ -f "/etc/redis/redis.conf" ]; then
        cp "/etc/redis/redis.conf" "${backup_path}/config/redis.conf"
    fi
    
    log_info "Configurations backup completed"
}

upload_to_s3() {
    if [ -z "${S3_BUCKET}" ]; then
        return 0
    fi
    
    log_info "Uploading backup to S3: s3://${S3_BUCKET}/${S3_PREFIX}"
    
    local backup_path="${BACKUP_DIR}/${BACKUP_NAME}"
    local archive_name="${BACKUP_NAME}.tar.gz"
    local archive_path="${BACKUP_DIR}/${archive_name}"
    local s3_path="s3://${S3_BUCKET}/${S3_PREFIX}${archive_name}"
    
    # Create compressed archive
    log_info "Creating archive for S3 upload..."
    if tar -czf "${archive_path}" -C "${BACKUP_DIR}" "${BACKUP_NAME}"; then
        local archive_size=$(du -sh "${archive_path}" | cut -f1)
        log_info "Archive created: ${archive_size}"
    else
        log_error "Failed to create archive"
        return 1
    fi
    
    # Upload to S3 with retry
    log_info "Uploading to S3..."
    if retry_command 3 5 upload_to_s3 "${archive_path}" "${s3_path}" "${S3_BUCKET}"; then
        log_success "Backup uploaded to S3 successfully"
        
        # Verify upload
        if aws s3 ls "${s3_path}" &> /dev/null; then
            log_success "S3 upload verified"
        else
            log_warn "S3 upload verification failed"
        fi
        
        # Clean up local archive
        rm -f "${archive_path}"
    else
        log_error "Failed to upload to S3 after retries"
        return 1
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (retention: ${RETENTION_DAYS} days)..."
    
    # Clean local backups
    find "${BACKUP_DIR}" -type d -name "ai-project-generator-backup-*" \
        -mtime +${RETENTION_DAYS} -exec rm -rf {} \; 2>/dev/null || true
    
    find "${BACKUP_DIR}" -type f -name "ai-project-generator-backup-*.tar.gz" \
        -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
    
    # Clean S3 backups
    if [ -n "${S3_BUCKET}" ] && command -v aws &> /dev/null; then
        local cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%Y%m%d 2>/dev/null || \
            date -v-${RETENTION_DAYS}d +%Y%m%d 2>/dev/null || echo "")
        
        if [ -n "${cutoff_date}" ]; then
            aws s3 ls "s3://${S3_BUCKET}/backups/" | \
                awk '{print $4}' | \
                while read backup_file; do
                    backup_date=$(echo "${backup_file}" | grep -oE '[0-9]{8}' | head -1)
                    if [ -n "${backup_date}" ] && [ "${backup_date}" -lt "${cutoff_date}" ]; then
                        log_info "Deleting old S3 backup: ${backup_file}"
                        aws s3 rm "s3://${S3_BUCKET}/backups/${backup_file}" || true
                    fi
                done
        fi
    fi
    
    log_info "Old backups cleaned up"
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "=========================================="
    log_info "Starting automated backup process"
    log_info "=========================================="
    
    validate_prerequisites
    backup_application_data || {
        log_error "Application data backup failed"
        exit 1
    }
    backup_logs
    backup_configurations
    
    if [ -n "${S3_BUCKET}" ]; then
        upload_to_s3 || {
            log_warn "S3 upload failed, but local backup succeeded"
        }
    fi
    
    cleanup_old_backups
    
    # Generate summary report
    local backup_size_human=$(calculate_directory_size "${BACKUP_DIR}/${BACKUP_NAME}")
    local duration_formatted=$(format_duration ${BACKUP_DURATION})
    
    log_info "=========================================="
    log_success "Backup process completed successfully"
    log_info "=========================================="
    log_info "Backup location: ${BACKUP_DIR}/${BACKUP_NAME}"
    log_info "Backup size: ${backup_size_human}"
    log_info "Duration: ${duration_formatted}"
    log_info "=========================================="
    
    # Send success notification if configured
    if [ -n "${SNS_TOPIC_ARN:-}" ]; then
        send_sns_alert "${SNS_TOPIC_ARN}" \
            "Backup Completed Successfully" \
            "Backup completed: ${BACKUP_NAME}\nSize: ${backup_size_human}\nDuration: ${duration_formatted}"
    fi
}

# Run main function
main "$@"

