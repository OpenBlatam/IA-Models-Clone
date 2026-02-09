#!/bin/bash

###############################################################################
# Automated Update Script for AI Project Generator
# Automatically updates the application with zero-downtime deployment
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANSIBLE_DIR="${SCRIPT_DIR}/../ansible"
APP_DIR="${APP_DIR:-/opt/ai-project-generator}"
GIT_REPO="${GIT_REPO:-}"
GIT_BRANCH="${GIT_BRANCH:-main}"
BACKUP_BEFORE_UPDATE="${BACKUP_BEFORE_UPDATE:-true}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-10}"
HEALTH_CHECK_DELAY="${HEALTH_CHECK_DELAY:-5}"
LOG_FILE="${LOG_FILE:-/var/log/automated-update.log}"

###############################################################################
# Helper Functions
###############################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Update failed with exit code: $exit_code"
        log_info "Attempting rollback..."
        rollback_update
    fi
    exit $exit_code
}

trap cleanup EXIT INT TERM

###############################################################################
# Validation
###############################################################################

validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check if Ansible is available
    if ! command -v ansible-playbook &> /dev/null; then
        log_error "Ansible not found. Please install Ansible first."
        exit 1
    fi
    
    # Check if application directory exists
    if [ ! -d "${APP_DIR}" ]; then
        log_error "Application directory not found: ${APP_DIR}"
        exit 1
    fi
    
    # Check if git repository is configured
    if [ -z "${GIT_REPO}" ] && [ ! -d "${APP_DIR}/.git" ]; then
        log_warn "Git repository not configured. Update will use existing code."
    fi
    
    log_info "Prerequisites validated."
}

###############################################################################
# Update Functions
###############################################################################

create_backup() {
    if [ "${BACKUP_BEFORE_UPDATE}" != "true" ]; then
        log_info "Skipping backup (BACKUP_BEFORE_UPDATE=false)"
        return 0
    fi
    
    log_info "Creating backup before update..."
    
    if [ -f "${SCRIPT_DIR}/automated_backup.sh" ]; then
        bash "${SCRIPT_DIR}/automated_backup.sh" || {
            log_warn "Backup failed, but continuing with update"
        }
    else
        log_warn "Backup script not found, skipping backup"
    fi
}

check_current_version() {
    log_info "Checking current application version..."
    
    local current_version="unknown"
    
    # Try to get version from git
    if [ -d "${APP_DIR}/.git" ]; then
        current_version=$(cd "${APP_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        log_info "Current Git commit: ${current_version}"
    fi
    
    # Try to get version from application
    if curl -s --max-time 5 "http://localhost:8020/api/v1/status" > /tmp/app_status.json 2>/dev/null; then
        current_version=$(jq -r '.version // "unknown"' /tmp/app_status.json 2>/dev/null || echo "unknown")
        log_info "Current application version: ${current_version}"
    fi
    
    echo "${current_version}" > /tmp/current_version.txt
}

pull_latest_code() {
    log_info "Pulling latest code..."
    
    if [ -n "${GIT_REPO}" ]; then
        log_info "Cloning/updating from: ${GIT_REPO} (branch: ${GIT_BRANCH})"
        
        if [ -d "${APP_DIR}/.git" ]; then
            cd "${APP_DIR}"
            git fetch origin
            git checkout "${GIT_BRANCH}"
            git pull origin "${GIT_BRANCH}"
        else
            git clone -b "${GIT_BRANCH}" "${GIT_REPO}" "${APP_DIR}.new"
            # Atomic replacement
            mv "${APP_DIR}" "${APP_DIR}.old"
            mv "${APP_DIR}.new" "${APP_DIR}"
        fi
    elif [ -d "${APP_DIR}/.git" ]; then
        log_info "Updating from existing Git repository..."
        cd "${APP_DIR}"
        git fetch origin
        git checkout "${GIT_BRANCH}"
        git pull origin "${GIT_BRANCH}"
    else
        log_warn "No Git repository found. Using existing code."
        return 0
    fi
    
    log_info "Code updated successfully"
}

deploy_with_ansible() {
    log_info "Deploying with Ansible..."
    
    cd "${ANSIBLE_DIR}" || exit 1
    
    # Update EC2 inventory
    if ! ansible-inventory -i inventory/ec2.ini --list > /dev/null 2>&1; then
        log_error "Failed to update EC2 inventory"
        return 1
    fi
    
    # Run update playbook
    ansible-playbook \
        -i inventory/ec2.ini \
        playbooks/update.yml \
        -e "git_branch=${GIT_BRANCH}" \
        --ask-become-pass || {
        log_error "Ansible deployment failed"
        return 1
    }
    
    log_info "Ansible deployment completed"
}

wait_for_health() {
    log_info "Waiting for application to become healthy..."
    
    local retries=0
    local max_retries=${HEALTH_CHECK_RETRIES}
    local delay=${HEALTH_CHECK_DELAY}
    
    while [ $retries -lt $max_retries ]; do
        if curl -f -s --max-time 5 "http://localhost:8020/health" > /dev/null 2>&1; then
            log_info "✅ Application is healthy"
            return 0
        fi
        
        retries=$((retries + 1))
        log_info "Health check attempt ${retries}/${max_retries}..."
        sleep $delay
    done
    
    log_error "Application failed to become healthy after ${max_retries} attempts"
    return 1
}

verify_update() {
    log_info "Verifying update..."
    
    # Check new version
    local new_version="unknown"
    if curl -s --max-time 5 "http://localhost:8020/api/v1/status" > /tmp/app_status_new.json 2>/dev/null; then
        new_version=$(jq -r '.version // "unknown"' /tmp/app_status_new.json 2>/dev/null || echo "unknown")
    fi
    
    local current_version
    current_version=$(cat /tmp/current_version.txt 2>/dev/null || echo "unknown")
    
    if [ "${new_version}" != "${current_version}" ] && [ "${new_version}" != "unknown" ]; then
        log_info "✅ Version updated: ${current_version} -> ${new_version}"
    else
        log_warn "Version unchanged or could not be determined"
    fi
    
    # Run health checks
    wait_for_health || return 1
    
    log_info "✅ Update verification completed"
}

rollback_update() {
    log_warn "Rolling back update..."
    
    # Stop new containers
    if [ -f "${APP_DIR}/docker-compose.yml" ]; then
        cd "${APP_DIR}"
        docker-compose down 2>/dev/null || true
    fi
    
    # Restore from backup if available
    if [ -d "${APP_DIR}.old" ]; then
        log_info "Restoring from previous version..."
        rm -rf "${APP_DIR}"
        mv "${APP_DIR}.old" "${APP_DIR}"
        
        # Restart application
        if [ -f "${APP_DIR}/docker-compose.yml" ]; then
            cd "${APP_DIR}"
            docker-compose up -d
        fi
    fi
    
    log_warn "Rollback completed"
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "Starting automated update process..."
    log_info "Repository: ${GIT_REPO:-local}"
    log_info "Branch: ${GIT_BRANCH}"
    
    validate_prerequisites
    create_backup
    check_current_version
    pull_latest_code
    deploy_with_ansible
    verify_update
    
    log_info "✅ Automated update completed successfully"
}

# Run main function
main "$@"

