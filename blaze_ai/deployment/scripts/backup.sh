#!/bin/bash

# Production Backup Script for Blaze AI
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/backups/blaze-ai"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30
COMPRESSION="gzip"

# Database configuration
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="blazeai"
DB_USER="blazeai"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking backup prerequisites..."
    
    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        log_info "Creating backup directory: $BACKUP_DIR"
        sudo mkdir -p $BACKUP_DIR
        sudo chown $USER:$USER $BACKUP_DIR
    fi
    
    # Check if required tools are installed
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump is not installed"
        exit 1
    fi
    
    if ! command -v redis-cli &> /dev/null; then
        log_error "redis-cli is not installed"
        exit 1
    fi
    
    if ! command -v tar &> /dev/null; then
        log_error "tar is not installed"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

backup_database() {
    log_info "Backing up PostgreSQL database..."
    
    BACKUP_FILE="$BACKUP_DIR/database_$DATE.sql"
    
    # Get database password from environment or prompt
    if [ -z "$DB_PASSWORD" ]; then
        read -s -p "Enter database password: " DB_PASSWORD
        echo
    fi
    
    # Create database backup
    PGPASSWORD=$DB_PASSWORD pg_dump \
        -h $DB_HOST \
        -p $DB_PORT \
        -U $DB_USER \
        -d $DB_NAME \
        --verbose \
        --clean \
        --if-exists \
        --create \
        --no-owner \
        --no-privileges \
        > $BACKUP_FILE
    
    if [ $? -eq 0 ]; then
        log_info "Database backup completed: $BACKUP_FILE"
        
        # Compress backup
        if [ "$COMPRESSION" = "gzip" ]; then
            gzip $BACKUP_FILE
            BACKUP_FILE="$BACKUP_FILE.gz"
        fi
        
        echo "$BACKUP_FILE" >> $BACKUP_DIR/backup_log.txt
    else
        log_error "Database backup failed"
        exit 1
    fi
}

backup_redis() {
    log_info "Backing up Redis data..."
    
    BACKUP_FILE="$BACKUP_DIR/redis_$DATE.rdb"
    
    # Get Redis password from environment or prompt
    if [ -z "$REDIS_PASSWORD" ]; then
        read -s -p "Enter Redis password: " REDIS_PASSWORD
        echo
    fi
    
    # Create Redis backup
    redis-cli -a $REDIS_PASSWORD --rdb $BACKUP_FILE
    
    if [ $? -eq 0 ]; then
        log_info "Redis backup completed: $BACKUP_FILE"
        
        # Compress backup
        if [ "$COMPRESSION" = "gzip" ]; then
            gzip $BACKUP_FILE
            BACKUP_FILE="$BACKUP_FILE.gz"
        fi
        
        echo "$BACKUP_FILE" >> $BACKUP_DIR/backup_log.txt
    else
        log_error "Redis backup failed"
        exit 1
    fi
}

backup_files() {
    log_info "Backing up application files..."
    
    BACKUP_FILE="$BACKUP_DIR/files_$DATE.tar"
    
    # Create file backup
    tar -czf $BACKUP_FILE \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='venv' \
        --exclude='.env' \
        --exclude='logs/*' \
        --exclude='temp/*' \
        --exclude='models/*' \
        --exclude='backups/*' \
        .
    
    if [ $? -eq 0 ]; then
        log_info "File backup completed: $BACKUP_FILE"
        echo "$BACKUP_FILE" >> $BACKUP_DIR/backup_log.txt
    else
        log_error "File backup failed"
        exit 1
    fi
}

backup_models() {
    log_info "Backing up AI models..."
    
    MODELS_DIR="models"
    if [ -d "$MODELS_DIR" ]; then
        BACKUP_FILE="$BACKUP_DIR/models_$DATE.tar"
        
        # Create models backup
        tar -czf $BACKUP_FILE $MODELS_DIR
        
        if [ $? -eq 0 ]; then
            log_info "Models backup completed: $BACKUP_FILE"
            echo "$BACKUP_FILE" >> $BACKUP_DIR/backup_log.txt
        else
            log_error "Models backup failed"
            exit 1
        fi
    else
        log_warn "Models directory not found, skipping models backup"
    fi
}

backup_logs() {
    log_info "Backing up application logs..."
    
    LOGS_DIR="logs"
    if [ -d "$LOGS_DIR" ]; then
        BACKUP_FILE="$BACKUP_DIR/logs_$DATE.tar"
        
        # Create logs backup (excluding current day)
        find $LOGS_DIR -name "*.log" -mtime +1 -exec tar -czf $BACKUP_FILE {} +
        
        if [ $? -eq 0 ]; then
            log_info "Logs backup completed: $BACKUP_FILE"
            echo "$BACKUP_FILE" >> $BACKUP_DIR/backup_log.txt
        else
            log_error "Logs backup failed"
            exit 1
        fi
    else
        log_warn "Logs directory not found, skipping logs backup"
    fi
}

create_backup_manifest() {
    log_info "Creating backup manifest..."
    
    MANIFEST_FILE="$BACKUP_DIR/manifest_$DATE.txt"
    
    cat > $MANIFEST_FILE << EOF
Blaze AI Backup Manifest
========================
Date: $(date)
Version: 2.0.0
Backup Type: Full

Components Backed Up:
- Database: PostgreSQL
- Cache: Redis
- Application Files
- AI Models
- Application Logs

Backup Files:
$(cat $BACKUP_DIR/backup_log.txt | tail -5)

System Information:
- Hostname: $(hostname)
- OS: $(uname -a)
- Disk Usage: $(df -h $BACKUP_DIR | tail -1)

Backup completed successfully at $(date)
EOF
    
    log_info "Backup manifest created: $MANIFEST_FILE"
}

cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    # Remove backups older than retention period
    find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    find $BACKUP_DIR -name "*.rdb.gz" -mtime +$RETENTION_DAYS -delete
    find $BACKUP_DIR -name "*.tar" -mtime +$RETENTION_DAYS -delete
    find $BACKUP_DIR -name "manifest_*.txt" -mtime +$RETENTION_DAYS -delete
    
    # Clean up backup log
    if [ -f "$BACKUP_DIR/backup_log.txt" ]; then
        # Keep only last 100 entries
        tail -100 $BACKUP_DIR/backup_log.txt > $BACKUP_DIR/backup_log_temp.txt
        mv $BACKUP_DIR/backup_log_temp.txt $BACKUP_DIR/backup_log.txt
    fi
    
    log_info "Cleanup completed"
}

verify_backup() {
    log_info "Verifying backup integrity..."
    
    # Check if backup files exist and are not empty
    for file in $BACKUP_DIR/*$DATE*; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            log_info "✓ $file verified"
        else
            log_error "✗ $file verification failed"
            exit 1
        fi
    done
    
    log_info "Backup verification completed successfully"
}

show_backup_summary() {
    log_info "Backup Summary:"
    echo "Backup Directory: $BACKUP_DIR"
    echo "Backup Date: $DATE"
    echo "Total Size: $(du -sh $BACKUP_DIR | cut -f1)"
    echo "Files Created:"
    ls -la $BACKUP_DIR/*$DATE*
    echo ""
    echo "Next steps:"
    echo "1. Test backup restoration in a safe environment"
    echo "2. Store backup files in a secure off-site location"
    echo "3. Monitor backup directory space usage"
}

# Main backup logic
main() {
    log_info "Starting Blaze AI production backup..."
    
    check_prerequisites
    
    # Create backup timestamp
    echo "$(date): Starting backup" >> $BACKUP_DIR/backup_log.txt
    
    # Perform backups
    backup_database
    backup_redis
    backup_files
    backup_models
    backup_logs
    
    # Create manifest and cleanup
    create_backup_manifest
    cleanup_old_backups
    verify_backup
    
    # Show summary
    show_backup_summary
    
    log_info "Backup completed successfully!"
}

# Run main function
main "$@"
