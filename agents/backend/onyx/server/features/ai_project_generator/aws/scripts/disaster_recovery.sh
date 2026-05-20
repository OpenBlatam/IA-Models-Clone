#!/bin/bash

###############################################################################
# Disaster Recovery Script for AI Project Generator
# Automated disaster recovery and failover procedures
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh" 2>/dev/null || {
    echo "Error: common_functions.sh not found" >&2
    exit 1
}

# Configuration
LOG_FILE="${LOG_FILE:-/var/log/disaster-recovery.log}"
BACKUP_DIR="${BACKUP_DIR:-/opt/backups/ai-project-generator}"
S3_BUCKET="${S3_BUCKET:-}"
APP_DIR="${APP_DIR:-/opt/ai-project-generator}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-10}"
HEALTH_CHECK_DELAY="${HEALTH_CHECK_DELAY:-5}"

# Recovery options
RECOVERY_MODE="${RECOVERY_MODE:-restore}" # restore, failover, rebuild
RESTORE_FROM_S3="${RESTORE_FROM_S3:-true}"
FAILOVER_REGION="${FAILOVER_REGION:-us-west-2}"

###############################################################################
# Recovery Functions
###############################################################################

detect_disaster() {
    log_info "Detecting disaster scenario..."
    
    local issues=0
    
    # Check application health
    if ! check_url "http://localhost:8020/health" 10 3; then
        log_error "Application health check failed"
        issues=$((issues + 1))
    fi
    
    # Check critical services
    if ! systemctl is-active --quiet nginx; then
        log_error "Nginx service is down"
        issues=$((issues + 1))
    fi
    
    if ! systemctl is-active --quiet redis-server; then
        log_error "Redis service is down"
        issues=$((issues + 1))
    fi
    
    # Check disk space
    local disk_usage
    disk_usage=$(get_disk_usage "/")
    if [ "${disk_usage}" -gt 95 ]; then
        log_error "Critical disk usage: ${disk_usage}%"
        issues=$((issues + 1))
    fi
    
    # Check application directory
    if [ ! -d "${APP_DIR}" ]; then
        log_error "Application directory missing"
        issues=$((issues + 1))
    fi
    
    if [ $issues -gt 0 ]; then
        log_error "Disaster detected: ${issues} critical issues found"
        return 1
    fi
    
    log_success "No disaster detected"
    return 0
}

restore_from_backup() {
    log_info "Restoring from backup..."
    
    local latest_backup
    latest_backup=$(find "${BACKUP_DIR}" -type d -name "ai-project-generator-backup-*" | \
        sort -r | head -1)
    
    if [ -z "${latest_backup}" ]; then
        log_error "No local backup found"
        return 1
    fi
    
    log_info "Found backup: ${latest_backup}"
    
    # Stop application
    if [ -f "${APP_DIR}/docker-compose.yml" ]; then
        cd "${APP_DIR}"
        docker-compose down || true
    fi
    
    # Restore application files
    if [ -f "${latest_backup}/application.tar.gz" ]; then
        log_info "Restoring application files..."
        rm -rf "${APP_DIR}"
        mkdir -p "${APP_DIR}"
        tar -xzf "${latest_backup}/application.tar.gz" -C "$(dirname ${APP_DIR})" || {
            log_error "Failed to restore application files"
            return 1
        }
    fi
    
    # Restore environment file
    if [ -f "${latest_backup}/.env" ]; then
        cp "${latest_backup}/.env" "${APP_DIR}/.env"
        log_info "Environment file restored"
    fi
    
    # Restore database
    if [ -f "${latest_backup}/database.sql" ] && check_command "psql"; then
        log_info "Restoring database..."
        psql -U postgres ai_project_generator < "${latest_backup}/database.sql" || {
            log_warn "Database restore had warnings"
        }
    fi
    
    # Restore Redis
    if [ -f "${latest_backup}/redis.rdb" ] && check_command "redis-cli"; then
        log_info "Restoring Redis data..."
        systemctl stop redis-server || true
        cp "${latest_backup}/redis.rdb" /var/lib/redis/dump.rdb
        systemctl start redis-server
    fi
    
    # Restart application
    if [ -f "${APP_DIR}/docker-compose.yml" ]; then
        cd "${APP_DIR}"
        docker-compose up -d || {
            log_error "Failed to start application"
            return 1
        }
    fi
    
    log_success "Restore from backup completed"
    return 0
}

restore_from_s3() {
    if [ -z "${S3_BUCKET}" ] || [ "${RESTORE_FROM_S3}" != "true" ]; then
        return 0
    fi
    
    log_info "Restoring from S3 backup..."
    
    if ! check_aws_credentials; then
        log_error "AWS credentials not configured"
        return 1
    fi
    
    # Find latest backup in S3
    local latest_backup
    latest_backup=$(aws s3 ls "s3://${S3_BUCKET}/backups/" | \
        grep "ai-project-generator-backup-" | \
        sort -r | head -1 | awk '{print $4}')
    
    if [ -z "${latest_backup}" ]; then
        log_error "No S3 backup found"
        return 1
    fi
    
    log_info "Found S3 backup: ${latest_backup}"
    
    # Download backup
    local temp_dir="/tmp/dr-restore-$(date +%s)"
    mkdir -p "${temp_dir}"
    
    aws s3 cp "s3://${S3_BUCKET}/backups/${latest_backup}" \
        "${temp_dir}/${latest_backup}" || {
        log_error "Failed to download backup from S3"
        return 1
    }
    
    # Extract backup
    tar -xzf "${temp_dir}/${latest_backup}" -C "${temp_dir}" || {
        log_error "Failed to extract backup"
        return 1
    }
    
    # Restore from extracted backup
    local backup_path="${temp_dir}/$(basename ${latest_backup} .tar.gz)"
    BACKUP_DIR="${temp_dir}"
    restore_from_backup
    
    # Cleanup
    rm -rf "${temp_dir}"
    
    log_success "Restore from S3 completed"
    return 0
}

rebuild_application() {
    log_info "Rebuilding application from scratch..."
    
    # Stop existing application
    if [ -f "${APP_DIR}/docker-compose.yml" ]; then
        cd "${APP_DIR}"
        docker-compose down || true
    fi
    
    # Clone from Git if repository is configured
    if [ -n "${GIT_REPO:-}" ]; then
        log_info "Cloning from Git repository..."
        rm -rf "${APP_DIR}"
        git clone "${GIT_REPO}" "${APP_DIR}" || {
            log_error "Failed to clone repository"
            return 1
        }
    else
        log_error "Git repository not configured for rebuild"
        return 1
    fi
    
    # Install dependencies and start
    if [ -f "${APP_DIR}/docker-compose.yml" ]; then
        cd "${APP_DIR}"
        docker-compose build
        docker-compose up -d || {
            log_error "Failed to start application"
            return 1
        }
    fi
    
    log_success "Application rebuilt"
    return 0
}

verify_recovery() {
    log_info "Verifying recovery..."
    
    local retries=0
    local max_retries=${HEALTH_CHECK_RETRIES}
    local delay=${HEALTH_CHECK_DELAY}
    
    while [ $retries -lt $max_retries ]; do
        if check_url "http://localhost:8020/health" 10 1; then
            log_success "Recovery verified - application is healthy"
            return 0
        fi
        
        retries=$((retries + 1))
        log_info "Verification attempt ${retries}/${max_retries}..."
        sleep $delay
    done
    
    log_error "Recovery verification failed"
    return 1
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "=========================================="
    log_info "Starting Disaster Recovery Process"
    log_info "=========================================="
    log_info "Recovery mode: ${RECOVERY_MODE}"
    
    # Detect if disaster has occurred
    if detect_disaster; then
        log_info "No disaster detected, exiting"
        exit 0
    fi
    
    # Send alert
    send_sns_alert "${SNS_TOPIC_ARN:-}" \
        "Disaster Detected - Recovery Initiated" \
        "Disaster recovery process started\nMode: ${RECOVERY_MODE}\nTime: $(date -Iseconds)"
    
    # Perform recovery based on mode
    case "${RECOVERY_MODE}" in
        restore)
            if [ "${RESTORE_FROM_S3}" = "true" ]; then
                restore_from_s3 || restore_from_backup
            else
                restore_from_backup
            fi
            ;;
        rebuild)
            rebuild_application
            ;;
        failover)
            log_info "Failover to ${FAILOVER_REGION} would be initiated here"
            # Implement failover logic
            ;;
        *)
            log_error "Unknown recovery mode: ${RECOVERY_MODE}"
            exit 1
            ;;
    esac
    
    # Verify recovery
    if verify_recovery; then
        send_sns_alert "${SNS_TOPIC_ARN:-}" \
            "Disaster Recovery Completed Successfully" \
            "Application has been recovered and is operational\nMode: ${RECOVERY_MODE}\nTime: $(date -Iseconds)"
        
        log_success "Disaster recovery completed successfully"
        exit 0
    else
        send_sns_alert "${SNS_TOPIC_ARN:-}" \
            "Disaster Recovery Failed" \
            "Recovery process completed but verification failed\nManual intervention required"
        
        log_error "Disaster recovery failed verification"
        exit 1
    fi
}

# Run main function
main "$@"

