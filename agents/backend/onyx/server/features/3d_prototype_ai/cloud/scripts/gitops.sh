#!/bin/bash
# GitOps script
# Implements GitOps practices

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly GITOPS_REPO="${GITOPS_REPO:-}"
readonly GITOPS_BRANCH="${GITOPS_BRANCH:-main}"
readonly GITOPS_PATH="${GITOPS_PATH:-manifests}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

GitOps deployment management.

COMMANDS:
    sync                Sync GitOps repository
    apply               Apply GitOps manifests
    diff                Show differences
    status              Show GitOps status
    rollback            Rollback to previous version

OPTIONS:
    -r, --repo URL          GitOps repository URL
    -b, --branch BRANCH     Git branch (default: main)
    -p, --path PATH         Manifests path (default: manifests)
    -h, --help              Show this help message

EXAMPLES:
    $0 sync --repo https://github.com/user/gitops-repo
    $0 apply
    $0 diff

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--repo)
                GITOPS_REPO="$2"
                shift 2
                ;;
            -b|--branch)
                GITOPS_BRANCH="$2"
                shift 2
                ;;
            -p|--path)
                GITOPS_PATH="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            sync|apply|diff|status|rollback)
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

# Sync GitOps repo
sync_repo() {
    local repo="${1}"
    local branch="${2}"
    local path="${3}"
    
    if [ -z "${repo}" ]; then
        error_exit 1 "GitOps repository URL is required"
    fi
    
    log_info "Syncing GitOps repository: ${repo}"
    
    local gitops_dir="/tmp/gitops-$$"
    mkdir -p "${gitops_dir}"
    
    # Clone or update repo
    if [ -d "${gitops_dir}/.git" ]; then
        cd "${gitops_dir}"
        git pull origin "${branch}" || log_warn "Git pull failed"
    else
        git clone -b "${branch}" "${repo}" "${gitops_dir}" || {
            log_error "Git clone failed"
            return 1
        }
    fi
    
    log_info "Repository synced to: ${gitops_dir}"
    echo "${gitops_dir}"
}

# Apply manifests
apply_manifests() {
    local repo="${1}"
    local branch="${2}"
    local path="${3}"
    
    log_info "Applying GitOps manifests..."
    
    local gitops_dir
    gitops_dir=$(sync_repo "${repo}" "${branch}" "${path}")
    
    local manifests_path="${gitops_dir}/${path}"
    
    if [ ! -d "${manifests_path}" ]; then
        log_error "Manifests path not found: ${manifests_path}"
        return 1
    fi
    
    # Apply Kubernetes manifests if kubectl available
    if command -v kubectl &> /dev/null; then
        log_info "Applying Kubernetes manifests..."
        kubectl apply -f "${manifests_path}" || {
            log_error "Manifest application failed"
            return 1
        }
    else
        log_info "Manifests ready at: ${manifests_path}"
        log_info "Apply manually or install kubectl"
    fi
    
    log_info "Manifests applied"
}

# Show differences
show_diff() {
    local repo="${1}"
    local branch="${2}"
    local path="${3}"
    
    log_info "Showing GitOps differences..."
    
    local gitops_dir
    gitops_dir=$(sync_repo "${repo}" "${branch}" "${path}")
    
    cd "${gitops_dir}"
    git diff HEAD~1 HEAD -- "${path}" || log_info "No differences found"
}

# Show status
show_status() {
    local repo="${1}"
    local branch="${2}"
    
    if [ -z "${repo}" ]; then
        error_exit 1 "GitOps repository URL is required"
    fi
    
    log_info "GitOps status:"
    
    local gitops_dir="/tmp/gitops-status-$$"
    git clone -b "${branch}" "${repo}" "${gitops_dir}" 2>/dev/null || {
        log_warn "Could not clone repository"
        return 1
    }
    
    cd "${gitops_dir}"
    echo "Repository: ${repo}"
    echo "Branch: ${branch}"
    echo "Latest commit: $(git log -1 --oneline)"
    echo "Last updated: $(git log -1 --format=%cd)"
    
    rm -rf "${gitops_dir}"
}

# Rollback
rollback_gitops() {
    local repo="${1}"
    local branch="${2}"
    local path="${3}"
    
    log_warn "Rolling back GitOps deployment..."
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Rollback cancelled"
        return 0
    fi
    
    log_info "Rolling back to previous version..."
    
    local gitops_dir
    gitops_dir=$(sync_repo "${repo}" "${branch}" "${path}")
    
    cd "${gitops_dir}"
    git checkout HEAD~1 -- "${path}" || {
        log_error "Rollback failed"
        return 1
    }
    
    # Apply previous version
    apply_manifests "${repo}" "${branch}" "${path}"
    
    log_info "Rollback completed"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        sync)
            if [ -z "${GITOPS_REPO}" ]; then
                error_exit 1 "GitOps repository URL is required"
            fi
            sync_repo "${GITOPS_REPO}" "${GITOPS_BRANCH}" "${GITOPS_PATH}"
            ;;
        apply)
            if [ -z "${GITOPS_REPO}" ]; then
                error_exit 1 "GitOps repository URL is required"
            fi
            apply_manifests "${GITOPS_REPO}" "${GITOPS_BRANCH}" "${GITOPS_PATH}"
            ;;
        diff)
            if [ -z "${GITOPS_REPO}" ]; then
                error_exit 1 "GitOps repository URL is required"
            fi
            show_diff "${GITOPS_REPO}" "${GITOPS_BRANCH}" "${GITOPS_PATH}"
            ;;
        status)
            show_status "${GITOPS_REPO}" "${GITOPS_BRANCH}"
            ;;
        rollback)
            if [ -z "${GITOPS_REPO}" ]; then
                error_exit 1 "GitOps repository URL is required"
            fi
            rollback_gitops "${GITOPS_REPO}" "${GITOPS_BRANCH}" "${GITOPS_PATH}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


