#!/bin/bash
# Rollback script for 3D Prototype AI deployment
# Restores previous version from backup

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."
BACKUP_DIR="${BACKUP_DIR:-${CLOUD_DIR}/backups}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [BACKUP_FILE]

Rollback 3D Prototype AI deployment to a previous version.

OPTIONS:
    -i, --ip IP              Instance IP address (required)
    -k, --key-path PATH      Path to SSH private key (required)
    -b, --backup-dir DIR     Backup directory (default: ./backups)
    -l, --list               List available backups
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem
    $0 --list
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem backups/remote_1.2.3.4_20240101_120000.tar.gz

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
            -b|--backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -l|--list)
                LIST_BACKUPS="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                if [ -z "${BACKUP_FILE:-}" ]; then
                    BACKUP_FILE="$1"
                else
                    log_error "Unknown option: $1"
                    usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
}

# List available backups
list_backups() {
    log_info "Available backups:"
    echo ""
    
    if [ ! -d "${BACKUP_DIR}" ]; then
        log_error "Backup directory not found: ${BACKUP_DIR}"
        return 1
    fi
    
    local backups
    backups=$(find "${BACKUP_DIR}" -name "*.tar.gz" -type f | sort -r)
    
    if [ -z "${backups}" ]; then
        log_warn "No backups found"
        return 1
    fi
    
    local count=1
    echo "${backups}" | while read -r backup; do
        local size
        size=$(du -h "${backup}" | cut -f1)
        local date
        date=$(stat -c "%y" "${backup}" 2>/dev/null || stat -f "%Sm" "${backup}" 2>/dev/null || echo "unknown")
        echo "[${count}] ${backup}"
        echo "     Size: ${size}, Date: ${date}"
        echo ""
        count=$((count + 1))
    done
}

# Select backup interactively
select_backup() {
    if [ -n "${BACKUP_FILE:-}" ]; then
        if [ ! -f "${BACKUP_FILE}" ]; then
            error_exit 1 "Backup file not found: ${BACKUP_FILE}"
        fi
        echo "${BACKUP_FILE}"
        return 0
    fi
    
    log_info "Selecting latest backup..."
    
    local latest_backup
    latest_backup=$(find "${BACKUP_DIR}" -name "remote_${INSTANCE_IP}_*.tar.gz" -type f | sort -r | head -1)
    
    if [ -z "${latest_backup}" ]; then
        error_exit 1 "No backup found for instance ${INSTANCE_IP}"
    fi
    
    log_info "Selected backup: ${latest_backup}"
    echo "${latest_backup}"
}

# Restore backup
restore_backup() {
    local backup_file="${1}"
    
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "Instance IP and key path are required for restore"
    fi
    
    validate_file "${backup_file}" "Backup file"
    validate_file "${AWS_KEY_PATH}" "SSH private key"
    validate_ip "${INSTANCE_IP}"
    
    log_info "Starting rollback process..."
    log_warn "This will replace the current deployment with the backup version"
    
    # Confirm action
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    if [ "${confirm}" != "yes" ]; then
        log_info "Rollback cancelled"
        exit 0
    fi
    
    # Upload backup to remote
    log_info "Uploading backup to remote instance..."
    scp -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${backup_file}" \
        ubuntu@${INSTANCE_IP}:/tmp/restore_backup.tar.gz || error_exit 1 "Failed to upload backup"
    
    # Restore on remote
    log_info "Restoring backup on remote instance..."
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} << 'REMOTE_EOF'
set -e
cd /opt/3d-prototype-ai

# Stop services
if [ -f "docker-compose.yml" ]; then
    docker-compose down || true
fi

# Backup current state (just in case)
CURRENT_BACKUP="/tmp/current_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "${CURRENT_BACKUP}" storage/ logs/ .env 2>/dev/null || true

# Extract backup
tar -xzf /tmp/restore_backup.tar.gz -C /opt/3d-prototype-ai

# Restore permissions
sudo chown -R ubuntu:ubuntu /opt/3d-prototype-ai

# Restart services
if [ -f "docker-compose.yml" ]; then
    docker-compose up -d --build
    docker-compose ps
else
    sudo systemctl daemon-reload
    sudo systemctl restart 3d-prototype-ai || true
fi

# Clean up
rm -f /tmp/restore_backup.tar.gz

# Health check
sleep 5
if curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "✓ Rollback successful, application is healthy"
else
    echo "✗ Health check failed after rollback"
    exit 1
fi
REMOTE_EOF
    
    if [ $? -eq 0 ]; then
        log_info "Rollback completed successfully! ✓"
    else
        error_exit 1 "Rollback failed"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ "${LIST_BACKUPS:-false}" = "true" ]; then
        list_backups
        exit 0
    fi
    
    if [ -z "${INSTANCE_IP:-}" ]; then
        error_exit 1 "Instance IP is required. Use --ip option or --help for usage"
    fi
    
    local backup_file
    backup_file=$(select_backup)
    
    restore_backup "${backup_file}"
    
    log_info "Rollback process completed! 🎉"
}

main "$@"

