#!/bin/bash
# Backup script for 3D Prototype AI deployment
# Creates backups of application data and configuration

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly BACKUP_DIR="${BACKUP_DIR:-${CLOUD_DIR}/backups}"
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"
readonly RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Create backup of 3D Prototype AI deployment.

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH     Path to SSH private key
    -d, --backup-dir DIR     Backup directory (default: ./backups)
    -r, --retention DAYS     Retention days (default: 30)
    -l, --local-only         Only backup local files
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem
    $0 --local-only --backup-dir /backups

EOF
}

# Parse arguments
parse_args() {
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
            -d|--backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -l|--local-only)
                LOCAL_ONLY="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Create backup directory
setup_backup_dir() {
    mkdir -p "${BACKUP_DIR}"
    log_info "Backup directory: ${BACKUP_DIR}"
}

# Backup local configuration
backup_local_config() {
    log_info "Backing up local configuration..."
    
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="${BACKUP_DIR}/config_${timestamp}.tar.gz"
    
    tar -czf "${backup_file}" \
        -C "${CLOUD_DIR}" \
        terraform/ \
        ansible/ \
        cloudformation/ \
        scripts/ \
        user_data/ \
        .env 2>/dev/null || log_warn "Some files may not have been backed up"
    
    log_info "Local configuration backed up: ${backup_file}"
    echo "${backup_file}"
}

# Backup remote application data
backup_remote_data() {
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        log_warn "Instance IP or key path not provided, skipping remote backup"
        return 0
    fi
    
    log_info "Backing up remote application data..."
    
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="${BACKUP_DIR}/remote_${INSTANCE_IP}_${timestamp}.tar.gz"
    
    # Create backup on remote and download
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Create backup
BACKUP_FILE="/tmp/backup_\$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "\${BACKUP_FILE}" \
    storage/ \
    logs/ \
    docker-compose.yml \
    .env 2>/dev/null || true

echo "\${BACKUP_FILE}"
REMOTE_EOF
    
    # Download backup
    scp -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP}:/tmp/backup_*.tar.gz \
        "${backup_file}" 2>/dev/null || {
        log_warn "Failed to download remote backup"
        return 1
    }
    
    # Clean up remote backup
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} \
        "rm -f /tmp/backup_*.tar.gz" 2>/dev/null || true
    
    log_info "Remote data backed up: ${backup_file}"
    echo "${backup_file}"
}

# Clean old backups
clean_old_backups() {
    log_info "Cleaning backups older than ${RETENTION_DAYS} days..."
    
    local deleted_count
    deleted_count=$(find "${BACKUP_DIR}" -name "*.tar.gz" -type f -mtime +${RETENTION_DAYS} -delete -print | wc -l)
    
    if [ "${deleted_count}" -gt 0 ]; then
        log_info "Deleted ${deleted_count} old backup(s)"
    else
        log_info "No old backups to clean"
    fi
}

# List backups
list_backups() {
    log_info "Available backups:"
    echo ""
    
    if [ -d "${BACKUP_DIR}" ] && [ "$(ls -A ${BACKUP_DIR}/*.tar.gz 2>/dev/null)" ]; then
        ls -lh "${BACKUP_DIR}"/*.tar.gz | awk '{print $9, $5, $6, $7, $8}'
    else
        log_warn "No backups found"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    log_info "Starting backup process..."
    
    setup_backup_dir
    
    # Backup local configuration
    backup_local_config
    
    # Backup remote data if not local-only
    if [ "${LOCAL_ONLY:-false}" != "true" ]; then
        backup_remote_data || log_warn "Remote backup failed, continuing..."
    fi
    
    # Clean old backups
    clean_old_backups
    
    # List backups
    echo ""
    list_backups
    
    log_info "Backup process completed! ✓"
}

main "$@"

