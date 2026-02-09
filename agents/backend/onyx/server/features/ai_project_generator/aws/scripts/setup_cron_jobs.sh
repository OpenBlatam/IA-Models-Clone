#!/bin/bash

###############################################################################
# Setup Cron Jobs Script
# Configures automated cron jobs for backups, monitoring, and updates
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_USER="${CRON_USER:-ubuntu}"

###############################################################################
# Helper Functions
###############################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

###############################################################################
# Cron Job Setup
###############################################################################

setup_backup_cron() {
    log_info "Setting up automated backup cron job..."
    
    local backup_script="${SCRIPT_DIR}/automated_backup.sh"
    local cron_schedule="${BACKUP_SCHEDULE:-0 2 * * *}"  # Default: 2 AM daily
    
    # Make script executable
    chmod +x "${backup_script}"
    
    # Add cron job (avoid duplicates)
    (crontab -u "${CRON_USER}" -l 2>/dev/null | \
        grep -v "automated_backup.sh" | \
        cat - <(echo "${cron_schedule} ${backup_script} >> /var/log/backup-cron.log 2>&1")) | \
        crontab -u "${CRON_USER}" -
    
    log_info "✅ Backup cron job configured: ${cron_schedule}"
}

setup_monitoring_cron() {
    log_info "Setting up automated monitoring cron job..."
    
    local monitoring_script="${SCRIPT_DIR}/automated_monitoring.sh"
    local cron_schedule="${MONITORING_SCHEDULE:-*/5 * * * *}"  # Default: Every 5 minutes
    
    # Make script executable
    chmod +x "${monitoring_script}"
    
    # Add cron job (avoid duplicates)
    (crontab -u "${CRON_USER}" -l 2>/dev/null | \
        grep -v "automated_monitoring.sh" | \
        cat - <(echo "${cron_schedule} ${monitoring_script} >> /var/log/monitoring-cron.log 2>&1")) | \
        crontab -u "${CRON_USER}" -
    
    log_info "✅ Monitoring cron job configured: ${cron_schedule}"
}

setup_update_cron() {
    log_info "Setting up automated update cron job..."
    
    local update_script="${SCRIPT_DIR}/automated_update.sh"
    local cron_schedule="${UPDATE_SCHEDULE:-0 3 * * 0}"  # Default: 3 AM on Sundays
    
    # Make script executable
    chmod +x "${update_script}"
    
    # Add cron job (avoid duplicates)
    (crontab -u "${CRON_USER}" -l 2>/dev/null | \
        grep -v "automated_update.sh" | \
        cat - <(echo "${cron_schedule} ${update_script} >> /var/log/update-cron.log 2>&1")) | \
        crontab -u "${CRON_USER}" -
    
    log_info "✅ Update cron job configured: ${cron_schedule}"
}

setup_log_rotation() {
    log_info "Setting up log rotation..."
    
    cat > /etc/logrotate.d/ai-project-generator <<EOF
/var/log/backup-cron.log
/var/log/monitoring-cron.log
/var/log/update-cron.log
/var/log/backup.log
/var/log/monitoring.log
/var/log/automated-update.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ${CRON_USER} ${CRON_USER}
}
EOF
    
    log_info "✅ Log rotation configured"
}

list_cron_jobs() {
    log_info "Current cron jobs for user ${CRON_USER}:"
    echo ""
    crontab -u "${CRON_USER}" -l 2>/dev/null || echo "No cron jobs configured"
    echo ""
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "Setting up automated cron jobs..."
    
    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root or with sudo"
        exit 1
    fi
    
    # Create log directory
    mkdir -p /var/log
    touch /var/log/backup-cron.log
    touch /var/log/monitoring-cron.log
    touch /var/log/update-cron.log
    chown "${CRON_USER}:${CRON_USER}" /var/log/*-cron.log
    
    # Setup cron jobs
    setup_backup_cron
    setup_monitoring_cron
    setup_update_cron
    setup_log_rotation
    
    # Display configured jobs
    list_cron_jobs
    
    log_info "✅ Cron jobs setup completed"
    log_info "Logs location: /var/log/*-cron.log"
}

# Run main function
main "$@"

