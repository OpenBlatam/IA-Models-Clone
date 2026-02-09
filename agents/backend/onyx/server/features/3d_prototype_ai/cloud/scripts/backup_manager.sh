#!/bin/bash
# Backup management script
# Manages application backups with retention policies

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"
readonly BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
readonly BACKUP_DIR="${BACKUP_DIR:-/tmp/backups}"
readonly S3_BUCKET="${S3_BACKUP_BUCKET:-}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Manage application backups.

COMMANDS:
    create              Create a new backup
    list                List all backups
    restore BACKUP      Restore from backup
    delete BACKUP       Delete a backup
    cleanup             Clean up old backups
    sync                Sync backups to S3
    status              Show backup status

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -r, --retention DAYS     Retention days (default: 7)
    -d, --backup-dir DIR     Backup directory (default: /tmp/backups)
    -b, --s3-bucket BUCKET   S3 bucket for backups
    -h, --help               Show this help message

EXAMPLES:
    $0 create --ip 1.2.3.4
    $0 list --ip 1.2.3.4
    $0 restore backup_20240101_120000.tar.gz --ip 1.2.3.4
    $0 cleanup --retention 30

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            -k|--key-path)
                AWS_KEY_PATH="$2"
                shift 2
                ;;
            -r|--retention)
                BACKUP_RETENTION_DAYS="$2"
                shift 2
                ;;
            -d|--backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -b|--s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            create|list|restore|delete|cleanup|sync|status)
                COMMAND="$1"
                if [ "$COMMAND" = "restore" ] || [ "$COMMAND" = "delete" ]; then
                    BACKUP_NAME="$2"
                    shift 2
                else
                    shift
                fi
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Create backup
create_backup() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="backup_${timestamp}.tar.gz"
    
    log_info "Creating backup: ${backup_name}"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Create backup
BACKUP_FILE="${backup_dir}/${backup_name}"
mkdir -p "${backup_dir}"

if [ -f "docker-compose.yml" ]; then
  echo "Creating Docker backup..."
  docker-compose exec -T api tar -czf "\${BACKUP_FILE}" . 2>/dev/null || \
  tar -czf "\${BACKUP_FILE}" storage/ logs/ .env docker-compose.yml 2>/dev/null || true
else
  echo "Creating standard backup..."
  tar -czf "\${BACKUP_FILE}" storage/ logs/ .env 2>/dev/null || true
fi

if [ -f "\${BACKUP_FILE}" ]; then
  BACKUP_SIZE=\$(du -h "\${BACKUP_FILE}" | cut -f1)
  echo "✅ Backup created: \${BACKUP_FILE} (Size: \${BACKUP_SIZE})"
  echo "\${BACKUP_FILE}"
else
  echo "❌ Backup creation failed"
  exit 1
fi
REMOTE_EOF
}

# List backups
list_backups() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Listing backups in ${backup_dir}..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
if [ -d "${backup_dir}" ]; then
  echo "Backups:"
  ls -lh "${backup_dir}"/*.tar.gz 2>/dev/null | awk '{print \$9, "(" \$5 ")"}' || echo "No backups found"
else
  echo "Backup directory does not exist"
fi
REMOTE_EOF
}

# Restore backup
restore_backup() {
    local ip="${1}"
    local key_path="${2}"
    local backup_name="${3}"
    local backup_dir="${4}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    if [ -z "${backup_name}" ]; then
        error_exit 1 "Backup name is required"
    fi
    
    local backup_path="${backup_dir}/${backup_name}"
    
    log_warn "This will restore from backup: ${backup_name}"
    log_warn "Current application will be replaced!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Restore cancelled"
        return 0
    fi
    
    log_info "Restoring from backup: ${backup_name}"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

if [ ! -f "${backup_path}" ]; then
  echo "❌ Backup file not found: ${backup_path}"
  exit 1
fi

# Stop application
if [ -f "docker-compose.yml" ]; then
  docker-compose down || true
fi

# Restore
echo "Extracting backup..."
tar -xzf "${backup_path}" -C /opt/3d-prototype-ai

# Restart application
if [ -f "docker-compose.yml" ]; then
  docker-compose up -d
  sleep 5
  docker-compose ps
else
  sudo systemctl restart 3d-prototype-ai || true
fi

# Verify
sleep 3
if curl -f http://localhost:8030/health > /dev/null 2>&1; then
  echo "✅ Restore completed successfully"
else
  echo "⚠️ Restore completed but health check failed"
fi
REMOTE_EOF
}

# Cleanup old backups
cleanup_backups() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    local retention_days="${4}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Cleaning up backups older than ${retention_days} days..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
if [ -d "${backup_dir}" ]; then
  find "${backup_dir}" -name "*.tar.gz" -type f -mtime +${retention_days} -delete
  REMAINING=\$(ls -1 "${backup_dir}"/*.tar.gz 2>/dev/null | wc -l)
  echo "✅ Cleanup completed. Remaining backups: \${REMAINING}"
else
  echo "Backup directory does not exist"
fi
REMOTE_EOF
}

# Sync to S3
sync_to_s3() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    local s3_bucket="${4}"
    
    if [ -z "${s3_bucket}" ]; then
        error_exit 1 "S3_BUCKET is required for sync"
    fi
    
    log_info "Syncing backups to S3: ${s3_bucket}"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
if command -v aws &> /dev/null && [ -d "${backup_dir}" ]; then
  aws s3 sync "${backup_dir}" "s3://${s3_bucket}/backups/" --exclude "*" --include "*.tar.gz"
  echo "✅ Sync completed"
else
  echo "❌ AWS CLI not available or backup directory not found"
fi
REMOTE_EOF
}

# Show backup status
show_status() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
echo "Backup Status"
echo "============="
echo "Backup Directory: ${backup_dir}"
echo ""

if [ -d "${backup_dir}" ]; then
  BACKUP_COUNT=\$(ls -1 "${backup_dir}"/*.tar.gz 2>/dev/null | wc -l)
  TOTAL_SIZE=\$(du -sh "${backup_dir}" 2>/dev/null | cut -f1)
  
  echo "Total Backups: \${BACKUP_COUNT}"
  echo "Total Size: \${TOTAL_SIZE}"
  echo ""
  echo "Recent Backups:"
  ls -lht "${backup_dir}"/*.tar.gz 2>/dev/null | head -5 | awk '{print \$9, "(" \$5 ")"}'
else
  echo "Backup directory does not exist"
fi
REMOTE_EOF
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        create)
            create_backup "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_DIR}"
            ;;
        list)
            list_backups "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_DIR}"
            ;;
        restore)
            restore_backup "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_NAME}" "${BACKUP_DIR}"
            ;;
        delete)
            log_info "Delete functionality - implement as needed"
            ;;
        cleanup)
            cleanup_backups "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_DIR}" "${BACKUP_RETENTION_DAYS}"
            ;;
        sync)
            sync_to_s3 "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_DIR}" "${S3_BUCKET}"
            ;;
        status)
            show_status "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_DIR}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


