#!/bin/bash
# Cleanup script for deployment artifacts and temporary files
# Follows DevOps best practices for resource cleanup

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Cleanup deployment artifacts and temporary files.

OPTIONS:
    -a, --all              Clean all artifacts (terraform, backups, logs)
    -t, --terraform        Clean Terraform state and plans
    -b, --backups          Clean old backups
    -l, --logs             Clean log files
    -c, --cache            Clean cache and temporary files
    -d, --days DAYS        Retention days for backups (default: 7)
    -f, --force            Force cleanup without confirmation
    -h, --help             Show this help message

EXAMPLES:
    $0 --all --force
    $0 --terraform --backups --days 30
    $0 --logs --cache

EOF
}

# Parse arguments
parse_args() {
    CLEAN_TERRAFORM=false
    CLEAN_BACKUPS=false
    CLEAN_LOGS=false
    CLEAN_CACHE=false
    CLEAN_ALL=false
    FORCE=false
    RETENTION_DAYS=7
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                CLEAN_ALL=true
                shift
                ;;
            -t|--terraform)
                CLEAN_TERRAFORM=true
                shift
                ;;
            -b|--backups)
                CLEAN_BACKUPS=true
                shift
                ;;
            -l|--logs)
                CLEAN_LOGS=true
                shift
                ;;
            -c|--cache)
                CLEAN_CACHE=true
                shift
                ;;
            -d|--days)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
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
    
    # If --all is specified, enable all cleanup options
    if [ "${CLEAN_ALL}" = "true" ]; then
        CLEAN_TERRAFORM=true
        CLEAN_BACKUPS=true
        CLEAN_LOGS=true
        CLEAN_CACHE=true
    fi
}

# Confirm cleanup
confirm_cleanup() {
    if [ "${FORCE}" = "true" ]; then
        return 0
    fi
    
    log_warn "This will clean up deployment artifacts."
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
}

# Clean Terraform artifacts
clean_terraform() {
    log_info "Cleaning Terraform artifacts..."
    
    local terraform_dir="${CLOUD_DIR}/terraform"
    
    if [ ! -d "${terraform_dir}" ]; then
        log_warn "Terraform directory not found, skipping..."
        return 0
    fi
    
    cd "${terraform_dir}"
    
    # Remove Terraform state and plans
    local removed=0
    for file in .terraform *.tfstate *.tfstate.* *.tfplan; do
        if [ -e "${file}" ] || [ -d "${file}" ]; then
            rm -rf "${file}"
            ((removed++))
        fi
    done
    
    if [ ${removed} -gt 0 ]; then
        log_info "Cleaned ${removed} Terraform artifact(s)"
    else
        log_info "No Terraform artifacts to clean"
    fi
    
    cd - > /dev/null
}

# Clean old backups
clean_backups() {
    log_info "Cleaning backups older than ${RETENTION_DAYS} days..."
    
    local backup_dir="${CLOUD_DIR}/backups"
    
    if [ ! -d "${backup_dir}" ]; then
        log_warn "Backup directory not found, skipping..."
        return 0
    fi
    
    local deleted_count
    deleted_count=$(find "${backup_dir}" -name "*.tar.gz" -type f -mtime +${RETENTION_DAYS} -delete -print | wc -l)
    
    if [ "${deleted_count}" -gt 0 ]; then
        log_info "Deleted ${deleted_count} old backup(s)"
    else
        log_info "No old backups to clean"
    fi
}

# Clean log files
clean_logs() {
    log_info "Cleaning log files..."
    
    local log_dirs=(
        "${CLOUD_DIR}/logs"
        "${CLOUD_DIR}/terraform"
        "${CLOUD_DIR}/ansible"
    )
    
    local removed=0
    for dir in "${log_dirs[@]}"; do
        if [ -d "${dir}" ]; then
            local count
            count=$(find "${dir}" -name "*.log" -type f -delete -print | wc -l)
            removed=$((removed + count))
        fi
    done
    
    if [ ${removed} -gt 0 ]; then
        log_info "Cleaned ${removed} log file(s)"
    else
        log_info "No log files to clean"
    fi
}

# Clean cache and temporary files
clean_cache() {
    log_info "Cleaning cache and temporary files..."
    
    local cache_dirs=(
        "${CLOUD_DIR}/.terraform"
        "${CLOUD_DIR}/.ansible"
        "${CLOUD_DIR}/tmp"
        "/tmp/deploy-*"
    )
    
    local removed=0
    for pattern in "${cache_dirs[@]}"; do
        for item in ${pattern}; do
            if [ -e "${item}" ] || [ -d "${item}" ]; then
                rm -rf "${item}"
                ((removed++))
            fi
        done
    done
    
    # Clean Python cache
    find "${CLOUD_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${CLOUD_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
    
    if [ ${removed} -gt 0 ]; then
        log_info "Cleaned cache and temporary files"
    else
        log_info "No cache files to clean"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    # Check if any cleanup option is selected
    if [ "${CLEAN_TERRAFORM}" = "false" ] && \
       [ "${CLEAN_BACKUPS}" = "false" ] && \
       [ "${CLEAN_LOGS}" = "false" ] && \
       [ "${CLEAN_CACHE}" = "false" ]; then
        log_error "No cleanup option selected. Use --help for usage."
        exit 1
    fi
    
    confirm_cleanup
    
    log_info "Starting cleanup process..."
    echo ""
    
    if [ "${CLEAN_TERRAFORM}" = "true" ]; then
        clean_terraform
        echo ""
    fi
    
    if [ "${CLEAN_BACKUPS}" = "true" ]; then
        clean_backups
        echo ""
    fi
    
    if [ "${CLEAN_LOGS}" = "true" ]; then
        clean_logs
        echo ""
    fi
    
    if [ "${CLEAN_CACHE}" = "true" ]; then
        clean_cache
        echo ""
    fi
    
    log_info "Cleanup completed! ✓"
}

main "$@"

