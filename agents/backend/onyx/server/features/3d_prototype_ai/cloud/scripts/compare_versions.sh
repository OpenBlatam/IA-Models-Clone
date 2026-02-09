#!/bin/bash
# Compare versions between local and remote
# Shows what will be deployed

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Compare local and remote versions.

OPTIONS:
    -i, --ip IP              Instance IP address (required)
    -k, --key-path PATH      Path to SSH private key (required)
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem

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
}

# Get file checksums from remote
get_remote_checksums() {
    local ip="${1}"
    local key_path="${2}"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
cd /opt/3d-prototype-ai
find . -type f -name "*.py" -exec md5sum {} \; 2>/dev/null | head -20
REMOTE_EOF
}

# Get local checksums
get_local_checksums() {
    local project_path="${1}"
    
    cd "${project_path}"
    find . -type f -name "*.py" -exec md5sum {} \; 2>/dev/null | head -20
}

# Compare versions
compare_versions() {
    log_info "Comparing local and remote versions..."
    
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    validate_file "${AWS_KEY_PATH}" "SSH private key"
    validate_ip "${INSTANCE_IP}"
    
    # Get git info
    local local_commit
    local_commit=$(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "unknown")
    local local_branch
    local_branch=$(cd "${PROJECT_ROOT}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    
    # Get remote info
    local remote_info
    remote_info=$(ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} \
        "cd /opt/3d-prototype-ai && git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo'" 2>/dev/null || echo "unknown")
    
    echo ""
    echo "${GREEN}==========================================${NC}"
    echo "${GREEN}Version Comparison${NC}"
    echo "${GREEN}==========================================${NC}"
    echo ""
    echo "Local:"
    echo "  Commit: ${local_commit}"
    echo "  Branch: ${local_branch}"
    echo ""
    echo "Remote:"
    if [ "${remote_info}" != "unknown" ] && [ "${remote_info}" != "not-a-git-repo" ]; then
        echo "  Commit: ${remote_info}"
    else
        echo "  Status: Not a git repository or unknown"
    fi
    echo ""
    
    # File comparison
    log_info "Comparing file checksums..."
    local local_checksums
    local_checksums=$(get_local_checksums "${PROJECT_ROOT}")
    local remote_checksums
    remote_checksums=$(get_remote_checksums "${INSTANCE_IP}" "${AWS_KEY_PATH}")
    
    echo "${GREEN}File Differences:${NC}"
    diff <(echo "${local_checksums}") <(echo "${remote_checksums}") || echo "Files differ or comparison unavailable"
    echo ""
    echo "${GREEN}==========================================${NC}"
}

# Main function
main() {
    parse_args "$@"
    compare_versions
}

main "$@"


