#!/bin/bash

# Production Backup Script for OS Content UGC Video Generator
# Handles database and file backups with compression and rotation

set -e

# Configuration
BACKUP_DIR="/backups"
DB_HOST="postgres"
DB_PORT="5432"
DB_NAME="os_content"
DB_USER="os_content_user"
DB_PASSWORD="${POSTGRES_PASSWORD}"
UPLOAD_DIR="/var/lib/os_content/uploads"
RETENTION_DAYS=30
COMPRESSION_LEVEL=9

# Timestamp for backup files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DATE=$(date +"%Y-%m-%d")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
    error "Backup directory $BACKUP_DIR does not exist"
    exit 1
fi

# Create backup subdirectories
DB_BACKUP_DIR="$BACKUP_DIR/database"
FILE_BACKUP_DIR="$BACKUP_DIR/files"
LOG_BACKUP_DIR="$BACKUP_DIR/logs"

mkdir -p "$DB_BACKUP_DIR" "$FILE_BACKUP_DIR" "$LOG_BACKUP_DIR"

log "Starting production backup at $TIMESTAMP"

# Function to cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    # Cleanup database backups
    find "$DB_BACKUP_DIR" -name "*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    
    # Cleanup file backups
    find "$FILE_BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    
    # Cleanup log backups
    find "$LOG_BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    
    log "Old backups cleanup completed"
}

# Function to backup database
backup_database() {
    log "Starting database backup"
    
    DB_BACKUP_FILE="$DB_BACKUP_DIR/os_content_db_$TIMESTAMP.sql"
    DB_BACKUP_COMPRESSED="$DB_BACKUP_FILE.gz"
    
    # Check database connectivity
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
        error "Database is not accessible"
        return 1
    fi
    
    # Create database dump
    log "Creating database dump..."
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --no-password \
        --clean \
        --if-exists \
        --create \
        --format=plain \
        --file="$DB_BACKUP_FILE"
    
    if [ $? -eq 0 ]; then
        log "Database dump created successfully"
        
        # Compress database backup
        log "Compressing database backup..."
        gzip -$COMPRESSION_LEVEL "$DB_BACKUP_FILE"
        
        if [ $? -eq 0 ]; then
            log "Database backup compressed: $DB_BACKUP_COMPRESSED"
            
            # Calculate backup size
            BACKUP_SIZE=$(du -h "$DB_BACKUP_COMPRESSED" | cut -f1)
            log "Database backup size: $BACKUP_SIZE"
            
            # Verify backup integrity
            log "Verifying database backup integrity..."
            if gzip -t "$DB_BACKUP_COMPRESSED"; then
                log "Database backup integrity verified"
            else
                error "Database backup integrity check failed"
                return 1
            fi
        else
            error "Failed to compress database backup"
            return 1
        fi
    else
        error "Failed to create database dump"
        return 1
    fi
    
    log "Database backup completed successfully"
}

# Function to backup files
backup_files() {
    log "Starting file backup"
    
    if [ ! -d "$UPLOAD_DIR" ]; then
        warn "Upload directory $UPLOAD_DIR does not exist, skipping file backup"
        return 0
    fi
    
    FILE_BACKUP_FILE="$FILE_BACKUP_DIR/os_content_files_$TIMESTAMP.tar"
    FILE_BACKUP_COMPRESSED="$FILE_BACKUP_FILE.gz"
    
    # Count files to backup
    FILE_COUNT=$(find "$UPLOAD_DIR" -type f | wc -l)
    log "Found $FILE_COUNT files to backup"
    
    if [ "$FILE_COUNT" -eq 0 ]; then
        log "No files to backup"
        return 0
    fi
    
    # Create file backup
    log "Creating file backup..."
    tar -cf "$FILE_BACKUP_FILE" -C "$(dirname "$UPLOAD_DIR")" "$(basename "$UPLOAD_DIR")" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log "File backup created successfully"
        
        # Compress file backup
        log "Compressing file backup..."
        gzip -$COMPRESSION_LEVEL "$FILE_BACKUP_FILE"
        
        if [ $? -eq 0 ]; then
            log "File backup compressed: $FILE_BACKUP_COMPRESSED"
            
            # Calculate backup size
            BACKUP_SIZE=$(du -h "$FILE_BACKUP_COMPRESSED" | cut -f1)
            log "File backup size: $BACKUP_SIZE"
            
            # Verify backup integrity
            log "Verifying file backup integrity..."
            if gzip -t "$FILE_BACKUP_COMPRESSED"; then
                log "File backup integrity verified"
            else
                error "File backup integrity check failed"
                return 1
            fi
        else
            error "Failed to compress file backup"
            return 1
        fi
    else
        error "Failed to create file backup"
        return 1
    fi
    
    log "File backup completed successfully"
}

# Function to backup logs
backup_logs() {
    log "Starting log backup"
    
    LOG_DIR="/var/log/os_content"
    LOG_BACKUP_FILE="$LOG_BACKUP_DIR/os_content_logs_$TIMESTAMP.tar"
    LOG_BACKUP_COMPRESSED="$LOG_BACKUP_FILE.gz"
    
    if [ ! -d "$LOG_DIR" ]; then
        warn "Log directory $LOG_DIR does not exist, skipping log backup"
        return 0
    fi
    
    # Count log files
    LOG_COUNT=$(find "$LOG_DIR" -name "*.log*" -type f | wc -l)
    log "Found $LOG_COUNT log files to backup"
    
    if [ "$LOG_COUNT" -eq 0 ]; then
        log "No log files to backup"
        return 0
    fi
    
    # Create log backup
    log "Creating log backup..."
    tar -cf "$LOG_BACKUP_FILE" -C "$(dirname "$LOG_DIR")" "$(basename "$LOG_DIR")" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log "Log backup created successfully"
        
        # Compress log backup
        log "Compressing log backup..."
        gzip -$COMPRESSION_LEVEL "$LOG_BACKUP_FILE"
        
        if [ $? -eq 0 ]; then
            log "Log backup compressed: $LOG_BACKUP_COMPRESSED"
            
            # Calculate backup size
            BACKUP_SIZE=$(du -h "$LOG_BACKUP_COMPRESSED" | cut -f1)
            log "Log backup size: $BACKUP_SIZE"
            
            # Verify backup integrity
            log "Verifying log backup integrity..."
            if gzip -t "$LOG_BACKUP_COMPRESSED"; then
                log "Log backup integrity verified"
            else
                error "Log backup integrity check failed"
                return 1
            fi
        else
            error "Failed to compress log backup"
            return 1
        fi
    else
        error "Failed to create log backup"
        return 1
    fi
    
    log "Log backup completed successfully"
}

# Function to create backup manifest
create_backup_manifest() {
    log "Creating backup manifest"
    
    MANIFEST_FILE="$BACKUP_DIR/backup_manifest_$TIMESTAMP.json"
    
    cat > "$MANIFEST_FILE" << EOF
{
    "backup_id": "$TIMESTAMP",
    "backup_date": "$BACKUP_DATE",
    "backup_time": "$(date -Iseconds)",
    "retention_days": $RETENTION_DAYS,
    "compression_level": $COMPRESSION_LEVEL,
    "backups": {
        "database": {
            "file": "database/os_content_db_$TIMESTAMP.sql.gz",
            "size": "$(du -h "$DB_BACKUP_DIR/os_content_db_$TIMESTAMP.sql.gz" 2>/dev/null | cut -f1 || echo 'N/A')",
            "status": "$([ -f "$DB_BACKUP_DIR/os_content_db_$TIMESTAMP.sql.gz" ] && echo 'success' || echo 'failed')"
        },
        "files": {
            "file": "files/os_content_files_$TIMESTAMP.tar.gz",
            "size": "$(du -h "$FILE_BACKUP_DIR/os_content_files_$TIMESTAMP.tar.gz" 2>/dev/null | cut -f1 || echo 'N/A')",
            "status": "$([ -f "$FILE_BACKUP_DIR/os_content_files_$TIMESTAMP.tar.gz" ] && echo 'success' || echo 'failed')"
        },
        "logs": {
            "file": "logs/os_content_logs_$TIMESTAMP.tar.gz",
            "size": "$(du -h "$LOG_BACKUP_DIR/os_content_logs_$TIMESTAMP.tar.gz" 2>/dev/null | cut -f1 || echo 'N/A')",
            "status": "$([ -f "$LOG_BACKUP_DIR/os_content_logs_$TIMESTAMP.tar.gz" ] && echo 'success' || echo 'failed')"
        }
    },
    "system_info": {
        "hostname": "$(hostname)",
        "disk_usage": "$(df -h "$BACKUP_DIR" | tail -1 | awk '{print $5}')",
        "backup_dir_size": "$(du -sh "$BACKUP_DIR" | cut -f1)"
    }
}
EOF
    
    log "Backup manifest created: $MANIFEST_FILE"
}

# Function to check disk space
check_disk_space() {
    log "Checking available disk space"
    
    AVAILABLE_SPACE=$(df "$BACKUP_DIR" | tail -1 | awk '{print $4}')
    AVAILABLE_SPACE_MB=$((AVAILABLE_SPACE / 1024))
    
    log "Available disk space: ${AVAILABLE_SPACE_MB}MB"
    
    if [ "$AVAILABLE_SPACE_MB" -lt 1000 ]; then
        warn "Low disk space available: ${AVAILABLE_SPACE_MB}MB"
        return 1
    fi
    
    return 0
}

# Main backup execution
main() {
    log "=== OS Content Production Backup Started ==="
    
    # Check disk space
    if ! check_disk_space; then
        error "Insufficient disk space for backup"
        exit 1
    fi
    
    # Cleanup old backups first
    cleanup_old_backups
    
    # Perform backups
    BACKUP_SUCCESS=true
    
    # Database backup
    if ! backup_database; then
        error "Database backup failed"
        BACKUP_SUCCESS=false
    fi
    
    # File backup
    if ! backup_files; then
        error "File backup failed"
        BACKUP_SUCCESS=false
    fi
    
    # Log backup
    if ! backup_logs; then
        error "Log backup failed"
        BACKUP_SUCCESS=false
    fi
    
    # Create backup manifest
    create_backup_manifest
    
    # Final status
    if [ "$BACKUP_SUCCESS" = true ]; then
        log "=== Production Backup Completed Successfully ==="
        
        # Calculate total backup size
        TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
        log "Total backup directory size: $TOTAL_SIZE"
        
        exit 0
    else
        error "=== Production Backup Completed with Errors ==="
        exit 1
    fi
}

# Execute main function
main "$@" 