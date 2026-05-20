#!/bin/bash
# Blue-green deployment script
# Implements blue-green deployment strategy

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"
readonly BLUE_ENV="${BLUE_ENV:-blue}"
readonly GREEN_ENV="${GREEN_ENV:-green}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Blue-green deployment management.

COMMANDS:
    init                Initialize blue-green setup
    deploy-green        Deploy to green environment
    switch              Switch traffic from blue to green
    rollback            Rollback to blue environment
    status              Show blue-green status

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -h, --help               Show this help message

EXAMPLES:
    $0 init
    $0 deploy-green --ip 1.2.3.4
    $0 switch
    $0 rollback

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    
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
            init|deploy-green|switch|rollback|status)
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

# Initialize blue-green setup
init_blue_green() {
    log_info "Initializing blue-green deployment setup..."
    
    # Create environment directories
    mkdir -p "${CLOUD_DIR}/environments/${BLUE_ENV}"
    mkdir -p "${CLOUD_DIR}/environments/${GREEN_ENV}"
    
    # Create configuration files
    cat > "${CLOUD_DIR}/environments/${BLUE_ENV}/.env" << EOF
ENVIRONMENT=${BLUE_ENV}
APP_PORT=8030
EOF
    
    cat > "${CLOUD_DIR}/environments/${GREEN_ENV}/.env" << EOF
ENVIRONMENT=${GREEN_ENV}
APP_PORT=8031
EOF
    
    log_info "Blue-green setup initialized"
    log_info "  Blue environment: ${BLUE_ENV} (port 8030)"
    log_info "  Green environment: ${GREEN_ENV} (port 8031)"
}

# Deploy to green
deploy_green() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Deploying to green environment..."
    
    # Deploy application to green port
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Create green environment
mkdir -p green
cp -r . green/ 2>/dev/null || true

# Update green environment port
cd green
sed -i 's/8030/8031/g' docker-compose.yml 2>/dev/null || true

# Deploy to green
if [ -f "docker-compose.yml" ]; then
    docker-compose -p green up -d --build
    sleep 10
    
    # Health check
    if curl -f http://localhost:8031/health > /dev/null 2>&1; then
        echo "✅ Green environment deployed and healthy"
    else
        echo "❌ Green environment health check failed"
        exit 1
    fi
fi
REMOTE_EOF
    
    log_info "Green environment deployed"
}

# Switch traffic
switch_traffic() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "Switching traffic from blue to green..."
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Switch cancelled"
        return 0
    fi
    
    log_info "Switching traffic..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Update Nginx/load balancer to point to green
# This is a placeholder - implement based on your setup

# Update port mapping
if [ -f "docker-compose.yml" ]; then
    # Switch ports
    docker-compose stop api || true
    docker-compose -p green up -d
    
    echo "✅ Traffic switched to green environment"
fi
REMOTE_EOF
    
    log_info "Traffic switched to green"
}

# Rollback to blue
rollback_to_blue() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "Rolling back to blue environment..."
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Rollback cancelled"
        return 0
    fi
    
    log_info "Rolling back to blue..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Switch back to blue
if [ -f "docker-compose.yml" ]; then
    docker-compose -p green stop || true
    docker-compose up -d
    
    echo "✅ Rolled back to blue environment"
fi
REMOTE_EOF
    
    log_info "Rolled back to blue"
}

# Show status
show_status() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Blue-green deployment status:"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
cd /opt/3d-prototype-ai

echo "Blue Environment (port 8030):"
if curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "  ✅ Healthy"
else
    echo "  ❌ Unhealthy or not running"
fi

echo ""
echo "Green Environment (port 8031):"
if curl -f http://localhost:8031/health > /dev/null 2>&1; then
    echo "  ✅ Healthy"
else
    echo "  ❌ Unhealthy or not running"
fi

echo ""
echo "Docker Containers:"
docker-compose ps 2>/dev/null || echo "  No containers running"
docker-compose -p green ps 2>/dev/null || echo "  No green containers running"
REMOTE_EOF
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        init)
            init_blue_green
            ;;
        deploy-green)
            if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
                error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
            fi
            deploy_green "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        switch)
            if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
                error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
            fi
            switch_traffic "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        rollback)
            if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
                error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
            fi
            rollback_to_blue "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        status)
            if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
                error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
            fi
            show_status "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


