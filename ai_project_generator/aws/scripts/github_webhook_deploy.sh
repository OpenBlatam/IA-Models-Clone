#!/bin/bash
# GitHub Webhook Deployment Script
# This script runs on EC2 instance and listens for deployment triggers
# Can be triggered via webhook or polling

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
LOG_FILE="/var/log/github-deploy.log"
LOCK_FILE="/var/run/github-deploy.lock"
GIT_REPO_URL="${GIT_REPO_URL:-}"
GIT_BRANCH="${GIT_BRANCH:-main}"
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
PROJECT_NAME="${PROJECT_NAME:-ai-project-generator}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_info() {
    log "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    log "${RED}[ERROR]${NC} $*"
}

# Cleanup function
cleanup() {
    if [ -f "$LOCK_FILE" ]; then
        rm -f "$LOCK_FILE"
    fi
}

# Trap signals
trap cleanup EXIT INT TERM

# Check if deployment is already running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_error "Deployment already running (PID: $pid)"
            return 1
        else
            log_warn "Stale lock file found, removing..."
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
    return 0
}

# Validate prerequisites
validate_prerequisites() {
    local missing_tools=()
    
    for tool in git curl jq; do
        if ! command -v "$tool" > /dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        return 1
    fi
    
    if [ -z "$GIT_REPO_URL" ]; then
        log_error "GIT_REPO_URL environment variable not set"
        return 1
    fi
    
    return 0
}

# Get latest commit from GitHub
get_latest_commit() {
    local repo_owner=$(echo "$GIT_REPO_URL" | sed -E 's|.*github.com[:/]([^/]+)/([^/]+)(\.git)?$|\1/\2|')
    local api_url="https://api.github.com/repos/$repo_owner/commits/$GIT_BRANCH"
    
    local commit_sha=$(curl -s "$api_url" | jq -r '.sha' 2>/dev/null || echo "")
    
    if [ -z "$commit_sha" ] || [ "$commit_sha" == "null" ]; then
        log_error "Failed to get latest commit from GitHub"
        return 1
    fi
    
    echo "$commit_sha"
}

# Check if deployment is needed
check_deployment_needed() {
    local latest_commit=$(get_latest_commit)
    local current_commit_file="/var/lib/$PROJECT_NAME/current_commit"
    
    if [ ! -f "$current_commit_file" ]; then
        log_info "No previous deployment found, deployment needed"
        echo "$latest_commit" > "$current_commit_file"
        return 0
    fi
    
    local current_commit=$(cat "$current_commit_file")
    
    if [ "$latest_commit" != "$current_commit" ]; then
        log_info "New commit detected: $latest_commit (current: $current_commit)"
        echo "$latest_commit" > "$current_commit_file"
        return 0
    else
        log_info "No new commits, deployment not needed"
        return 1
    fi
}

# Pull latest code
pull_latest_code() {
    local repo_dir="$PROJECT_ROOT"
    
    log_info "Pulling latest code from $GIT_BRANCH..."
    
    if [ ! -d "$repo_dir/.git" ]; then
        log_info "Cloning repository..."
        git clone -b "$GIT_BRANCH" "$GIT_REPO_URL" "$repo_dir" || return 1
    else
        log_info "Updating repository..."
        cd "$repo_dir"
        git fetch origin "$GIT_BRANCH" || return 1
        git reset --hard "origin/$GIT_BRANCH" || return 1
        git clean -fd || return 1
    fi
    
    log_info "Code updated successfully"
    return 0
}

# Run deployment
run_deployment() {
    log_info "Starting deployment..."
    
    local ansible_dir="$PROJECT_ROOT/agents/backend/onyx/server/features/ai_project_generator/aws/ansible"
    
    if [ ! -d "$ansible_dir" ]; then
        log_error "Ansible directory not found: $ansible_dir"
        return 1
    fi
    
    cd "$ansible_dir"
    
    # Create localhost inventory
    cat > /tmp/localhost-inventory.ini <<EOF
[local]
localhost ansible_connection=local
EOF
    
    # Run Ansible update playbook
    log_info "Running Ansible update playbook..."
    ansible-playbook -i /tmp/localhost-inventory.ini playbooks/update.yml \
        -e "project_name=$PROJECT_NAME" \
        -e "environment=production" \
        -e "git_branch=$GIT_BRANCH" \
        -e "git_commit=$(get_latest_commit)" || {
        log_error "Ansible deployment failed"
        return 1
    }
    
    log_info "Deployment completed successfully"
    return 0
}

# Health check
health_check() {
    local max_attempts=5
    local attempt=1
    local health_url="http://localhost/health"
    
    log_info "Performing health check..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            log_info "Health check passed"
            return 0
        else
            log_warn "Health check attempt $attempt/$max_attempts failed"
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Main deployment function
main() {
    log_info "=== GitHub Webhook Deployment Script ==="
    log_info "Project: $PROJECT_NAME"
    log_info "Branch: $GIT_BRANCH"
    
    # Check lock
    if ! check_lock; then
        exit 1
    fi
    
    # Validate prerequisites
    if ! validate_prerequisites; then
        exit 1
    fi
    
    # Check if deployment is needed
    if ! check_deployment_needed; then
        log_info "No deployment needed"
        exit 0
    fi
    
    # Pull latest code
    if ! pull_latest_code; then
        log_error "Failed to pull latest code"
        exit 1
    fi
    
    # Run deployment
    if ! run_deployment; then
        log_error "Deployment failed"
        exit 1
    fi
    
    # Health check
    if ! health_check; then
        log_warn "Health check failed, but deployment completed"
    fi
    
    log_info "=== Deployment completed successfully ==="
}

# Run main function
main "$@"

