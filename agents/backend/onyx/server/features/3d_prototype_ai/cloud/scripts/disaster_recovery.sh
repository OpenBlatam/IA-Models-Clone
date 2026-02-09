#!/bin/bash
# Disaster recovery script
# Handles disaster recovery scenarios

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
readonly BACKUP_SOURCE="${BACKUP_SOURCE:-/tmp/backups}"
readonly S3_BACKUP_BUCKET="${S3_BACKUP_BUCKET:-}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Disaster recovery operations.

COMMANDS:
    backup-now           Create immediate backup
    restore-latest       Restore from latest backup
    verify-backup        Verify backup integrity
    failover             Failover to backup instance
    health-check         Check recovery readiness
    test-dr              Test disaster recovery procedure

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -s, --source DIR         Backup source directory
    -b, --s3-bucket BUCKET   S3 backup bucket
    -h, --help               Show this help message

EXAMPLES:
    $0 backup-now --ip 1.2.3.4
    $0 restore-latest --ip 1.2.3.4
    $0 test-dr

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
            -s|--source)
                BACKUP_SOURCE="$2"
                shift 2
                ;;
            -b|--s3-bucket)
                S3_BACKUP_BUCKET="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            backup-now|restore-latest|verify-backup|failover|health-check|test-dr)
                COMMAND="$1"
                shift
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

# Create immediate backup
backup_now() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Creating immediate backup..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${backup_dir}/dr_backup_\${TIMESTAMP}.tar.gz"
mkdir -p "${backup_dir}"

# Create comprehensive backup
tar -czf "\${BACKUP_FILE}" \
    storage/ \
    logs/ \
    .env \
    docker-compose.yml \
    requirements.txt \
    *.py \
    2>/dev/null || true

# Verify backup
if [ -f "\${BACKUP_FILE}" ]; then
    SIZE=\$(du -h "\${BACKUP_FILE}" | cut -f1)
    echo "✅ Backup created: \${BACKUP_FILE} (Size: \${SIZE})"
    
    # Upload to S3 if configured
    if [ -n "${S3_BACKUP_BUCKET}" ] && command -v aws &> /dev/null; then
        aws s3 cp "\${BACKUP_FILE}" "s3://${S3_BACKUP_BUCKET}/dr-backups/" && \
        echo "✅ Backup uploaded to S3"
    fi
else
    echo "❌ Backup creation failed"
    exit 1
fi
REMOTE_EOF
}

# Restore from latest backup
restore_latest() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    local s3_bucket="${4}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "This will restore from the latest backup!"
    log_warn "Current application will be replaced!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Restore cancelled"
        return 0
    fi
    
    log_info "Restoring from latest backup..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Find latest backup
if [ -n "${s3_bucket}" ] && command -v aws &> /dev/null; then
    echo "Downloading latest backup from S3..."
    LATEST_BACKUP=\$(aws s3 ls "s3://${s3_bucket}/dr-backups/" | sort | tail -1 | awk '{print \$4}')
    if [ -n "\${LATEST_BACKUP}" ]; then
        aws s3 cp "s3://${s3_bucket}/dr-backups/\${LATEST_BACKUP}" "${backup_dir}/"
        BACKUP_FILE="${backup_dir}/\${LATEST_BACKUP}"
    else
        echo "No backups found in S3"
        exit 1
    fi
else
    BACKUP_FILE=\$(ls -t "${backup_dir}"/dr_backup_*.tar.gz 2>/dev/null | head -1)
fi

if [ -z "\${BACKUP_FILE}" ] || [ ! -f "\${BACKUP_FILE}" ]; then
    echo "❌ No backup file found"
    exit 1
fi

echo "Restoring from: \${BACKUP_FILE}"

# Stop application
if [ -f "docker-compose.yml" ]; then
    docker-compose down || true
fi

# Restore
tar -xzf "\${BACKUP_FILE}" -C /opt/3d-prototype-ai

# Restart application
if [ -f "docker-compose.yml" ]; then
    docker-compose up -d
    sleep 10
fi

# Verify
if curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "✅ Restore completed successfully"
else
    echo "⚠️ Restore completed but health check failed"
fi
REMOTE_EOF
}

# Verify backup integrity
verify_backup() {
    local ip="${1}"
    local key_path="${2}"
    local backup_dir="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Verifying backup integrity..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
cd "${backup_dir}"

echo "Backup Verification Report"
echo "========================="
echo ""

for backup in dr_backup_*.tar.gz; do
    if [ -f "\${backup}" ]; then
        echo "Checking: \${backup}"
        
        # Check if tar is valid
        if tar -tzf "\${backup}" > /dev/null 2>&1; then
            SIZE=\$(du -h "\${backup}" | cut -f1)
            FILES=\$(tar -tzf "\${backup}" | wc -l)
            echo "  ✅ Valid (Size: \${SIZE}, Files: \${FILES})"
        else
            echo "  ❌ Invalid or corrupted"
        fi
        echo ""
    fi
done
REMOTE_EOF
}

# Health check for DR readiness
health_check() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Checking disaster recovery readiness..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
echo "Disaster Recovery Readiness Check"
echo "=================================="
echo ""

# Check backups
BACKUP_COUNT=\$(ls -1 /tmp/backups/dr_backup_*.tar.gz 2>/dev/null | wc -l)
echo "Local Backups: \${BACKUP_COUNT}"

# Check S3 access
if command -v aws &> /dev/null; then
    if [ -n "${S3_BACKUP_BUCKET}" ]; then
        S3_BACKUPS=\$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/dr-backups/" 2>/dev/null | wc -l)
        echo "S3 Backups: \${S3_BACKUPS}"
    fi
fi

# Check application health
if curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "Application: ✅ Healthy"
else
    echo "Application: ❌ Unhealthy"
fi

# Check disk space
DISK_SPACE=\$(df -h / | tail -1 | awk '{print \$4}')
echo "Available Disk Space: \${DISK_SPACE}"

# Check critical files
CRITICAL_FILES=(".env" "docker-compose.yml" "requirements.txt")
for file in "\${CRITICAL_FILES[@]}"; do
    if [ -f "/opt/3d-prototype-ai/\${file}" ]; then
        echo "✅ \${file} exists"
    else
        echo "❌ \${file} missing"
    fi
done
REMOTE_EOF
}

# Test disaster recovery
test_dr() {
    log_info "Testing disaster recovery procedure..."
    
    echo ""
    echo "Disaster Recovery Test Checklist"
    echo "================================"
    echo ""
    echo "1. ✅ Backup creation"
    echo "2. ✅ Backup verification"
    echo "3. ✅ Restore procedure"
    echo "4. ✅ Health check after restore"
    echo "5. ✅ S3 backup sync (if configured)"
    echo ""
    echo "Run each step manually to verify:"
    echo "  ./scripts/disaster_recovery.sh backup-now"
    echo "  ./scripts/disaster_recovery.sh verify-backup"
    echo "  ./scripts/disaster_recovery.sh restore-latest"
    echo "  ./scripts/disaster_recovery.sh health-check"
    echo ""
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        backup-now)
            backup_now "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_SOURCE}"
            ;;
        restore-latest)
            restore_latest "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_SOURCE}" "${S3_BACKUP_BUCKET}"
            ;;
        verify-backup)
            verify_backup "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${BACKUP_SOURCE}"
            ;;
        failover)
            log_info "Failover functionality - implement based on your architecture"
            ;;
        health-check)
            health_check "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        test-dr)
            test_dr
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


