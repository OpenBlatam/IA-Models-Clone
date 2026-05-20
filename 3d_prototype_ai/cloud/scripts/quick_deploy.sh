#!/bin/bash
# Quick deployment script
# Simplified deployment for rapid iterations

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

Quick deployment script for rapid iterations.

OPTIONS:
    -i, --ip IP              Instance IP address (required)
    -k, --key-path PATH      Path to SSH private key (required)
    -f, --force              Force deployment without backup
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem --force

EOF
}

# Parse arguments
parse_args() {
    FORCE=false
    
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
}

# Quick deployment
quick_deploy() {
    log_info "Starting quick deployment to ${INSTANCE_IP}..."
    
    # Validate
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    validate_file "${AWS_KEY_PATH}" "SSH private key"
    validate_ip "${INSTANCE_IP}"
    
    # Quick backup (skip if force)
    if [ "${FORCE}" != "true" ]; then
        log_info "Creating quick backup..."
        ssh -i "${AWS_KEY_PATH}" \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            ubuntu@${INSTANCE_IP} \
            "cd /opt/3d-prototype-ai && tar -czf /tmp/quick_backup_$(date +%Y%m%d_%H%M%S).tar.gz . 2>/dev/null || true" || true
    fi
    
    # Deploy files
    log_info "Deploying files..."
    rsync -avz \
        -e "ssh -i ${AWS_KEY_PATH} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.env' \
        --exclude='cloud/' \
        --exclude='*.md' \
        --exclude='tests/' \
        --exclude='.github/' \
        --delete \
        "${PROJECT_ROOT}/" \
        ubuntu@${INSTANCE_IP}:/opt/3d-prototype-ai/
    
    # Quick restart
    log_info "Restarting application..."
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} << 'REMOTE_EOF'
cd /opt/3d-prototype-ai
sudo chown -R ubuntu:ubuntu /opt/3d-prototype-ai

if [ -f "docker-compose.yml" ]; then
  docker-compose restart api || docker-compose up -d
else
  sudo systemctl restart 3d-prototype-ai || true
fi

sleep 3
curl -f http://localhost:8030/health > /dev/null 2>&1 && echo "✓ Healthy" || echo "✗ Unhealthy"
REMOTE_EOF
    
    log_info "Quick deployment completed! ✓"
}

# Main function
main() {
    parse_args "$@"
    quick_deploy
}

main "$@"


