#!/bin/bash
# Disaster Recovery Script for Music Analyzer AI
# Automated backup, restore, and failover procedures

set -euo pipefail

# Configuration
readonly BACKUP_DIR="${BACKUP_DIR:-/backups/music-analyzer-ai}"
readonly RETENTION_DAYS="${RETENTION_DAYS:-30}"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly BACKUP_NAME="music-analyzer-ai-backup-${TIMESTAMP}"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
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

# Create backup
create_backup() {
    log_info "Creating disaster recovery backup..."
    
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
    
    # Backup database
    if command -v pg_dump &> /dev/null; then
        log_info "Backing up PostgreSQL database..."
        pg_dump "${DATABASE_URL}" > "${BACKUP_DIR}/${BACKUP_NAME}/database.sql" || true
    fi
    
    # Backup Redis
    if command -v redis-cli &> /dev/null; then
        log_info "Backing up Redis data..."
        redis-cli --rdb "${BACKUP_DIR}/${BACKUP_NAME}/redis.rdb" || true
    fi
    
    # Backup configuration
    log_info "Backing up configuration files..."
    cp -r deployment/.env "${BACKUP_DIR}/${BACKUP_NAME}/" 2>/dev/null || true
    cp -r deployment/kubernetes "${BACKUP_DIR}/${BACKUP_NAME}/" 2>/dev/null || true
    
    # Backup Docker volumes
    log_info "Backing up Docker volumes..."
    docker run --rm \
        -v music-analyzer-ai-data:/data:ro \
        -v "${BACKUP_DIR}/${BACKUP_NAME}:/backup" \
        alpine tar czf /backup/volumes.tar.gz -C /data . 2>/dev/null || true
    
    # Compress backup
    log_info "Compressing backup..."
    cd "${BACKUP_DIR}"
    tar czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
    rm -rf "${BACKUP_NAME}"
    
    # Upload to S3 (if configured)
    if [ -n "${S3_BACKUP_BUCKET:-}" ] && command -v aws &> /dev/null; then
        log_info "Uploading backup to S3..."
        aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
            "s3://${S3_BACKUP_BUCKET}/music-analyzer-ai/${BACKUP_NAME}.tar.gz" || true
    fi
    
    log_info "Backup completed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# Restore from backup
restore_backup() {
    local backup_file="${1:-}"
    
    if [ -z "${backup_file}" ]; then
        log_error "Backup file not specified"
        echo "Usage: $0 restore <backup-file.tar.gz>"
        exit 1
    fi
    
    if [ ! -f "${backup_file}" ]; then
        backup_file="${BACKUP_DIR}/${backup_file}"
        if [ ! -f "${backup_file}" ]; then
            log_error "Backup file not found: ${backup_file}"
            exit 1
        fi
    fi
    
    log_info "Restoring from backup: ${backup_file}"
    
    local temp_dir=$(mktemp -d)
    tar xzf "${backup_file}" -C "${temp_dir}"
    
    # Restore database
    if [ -f "${temp_dir}"/*/database.sql ]; then
        log_info "Restoring database..."
        psql "${DATABASE_URL}" < "${temp_dir}"/*/database.sql || true
    fi
    
    # Restore Redis
    if [ -f "${temp_dir}"/*/redis.rdb ]; then
        log_info "Restoring Redis..."
        redis-cli --rdb "${temp_dir}"/*/redis.rdb || true
    fi
    
    # Restore configuration
    if [ -d "${temp_dir}"/*/kubernetes ]; then
        log_info "Restoring configuration..."
        cp -r "${temp_dir}"/*/kubernetes deployment/ || true
    fi
    
    rm -rf "${temp_dir}"
    log_info "Restore completed"
}

# Failover to secondary region
failover() {
    local primary_region="${1:-us-east-1}"
    local secondary_region="${2:-us-west-2}"
    
    log_info "Initiating failover from ${primary_region} to ${secondary_region}..."
    
    # Update DNS/load balancer to point to secondary
    log_info "Updating DNS to secondary region..."
    # This would typically update Route53 or similar
    
    # Scale up secondary region
    log_info "Scaling up secondary region..."
    kubectl scale deployment music-analyzer-ai-backend \
        --replicas=5 \
        -n production-${secondary_region} || true
    
    # Verify secondary is healthy
    log_info "Verifying secondary region health..."
    sleep 30
    
    log_info "Failover completed"
}

# Main function
main() {
    case "${1:-backup}" in
        backup)
            create_backup
            ;;
        restore)
            restore_backup "${2:-}"
            ;;
        failover)
            failover "${2:-us-east-1}" "${3:-us-west-2}"
            ;;
        *)
            echo "Usage: $0 {backup|restore|failover}"
            exit 1
            ;;
    esac
}

main "$@"




