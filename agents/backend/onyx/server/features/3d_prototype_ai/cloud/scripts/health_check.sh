#!/bin/bash
# Health check script for 3D Prototype AI deployment

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load configuration
if [ -f "${CLOUD_DIR}/.env" ]; then
    source "${CLOUD_DIR}/.env"
fi

# Default values
INSTANCE_IP="${INSTANCE_IP}"
APP_PORT="${APP_PORT:-8030}"
AWS_KEY_PATH="${AWS_KEY_PATH}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_health() {
    local url="${1}"
    local name="${2}"
    
    log_info "Checking ${name}..."
    
    if curl -f -s -o /dev/null -w "%{http_code}" "${url}" | grep -q "200"; then
        log_info "${name}: ✓ Healthy"
        return 0
    else
        log_error "${name}: ✗ Unhealthy"
        return 1
    fi
}

check_instance() {
    if [ -z "${INSTANCE_IP}" ]; then
        log_error "INSTANCE_IP not set"
        return 1
    fi
    
    log_info "Checking instance connectivity..."
    
    if ping -c 1 -W 2 "${INSTANCE_IP}" &> /dev/null; then
        log_info "Instance is reachable ✓"
        return 0
    else
        log_error "Instance is not reachable ✗"
        return 1
    fi
}

check_ssh() {
    if [ -z "${AWS_KEY_PATH}" ] || [ -z "${INSTANCE_IP}" ]; then
        log_warn "SSH check skipped (AWS_KEY_PATH or INSTANCE_IP not set)"
        return 0
    fi
    
    log_info "Checking SSH access..."
    
    if ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        ubuntu@${INSTANCE_IP} \
        "echo 'SSH OK'" &> /dev/null; then
        log_info "SSH access: ✓"
        return 0
    else
        log_error "SSH access: ✗"
        return 1
    fi
}

check_application() {
    local base_url="http://${INSTANCE_IP}"
    
    check_health "${base_url}:${APP_PORT}/health" "Application Health"
    check_health "${base_url}/health" "Nginx Health"
    check_health "${base_url}/docs" "API Documentation"
}

check_docker() {
    if [ -z "${AWS_KEY_PATH}" ] || [ -z "${INSTANCE_IP}" ]; then
        log_warn "Docker check skipped"
        return 0
    fi
    
    log_info "Checking Docker services..."
    
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        ubuntu@${INSTANCE_IP} << 'REMOTE_EOF'
cd /opt/3d-prototype-ai

if [ -f "docker-compose.yml" ]; then
    echo "Docker Compose services:"
    docker-compose ps
    
    echo ""
    echo "Docker logs (last 20 lines):"
    docker-compose logs --tail=20
else
    echo "Docker Compose not found, checking systemd service..."
    sudo systemctl status 3d-prototype-ai || true
fi
REMOTE_EOF
}

main() {
    log_info "Starting health check..."
    
    local all_healthy=true
    
    if ! check_instance; then
        all_healthy=false
    fi
    
    if ! check_ssh; then
        all_healthy=false
    fi
    
    if ! check_application; then
        all_healthy=false
    fi
    
    if [ "${CHECK_DOCKER}" = "true" ]; then
        check_docker
    fi
    
    echo ""
    if [ "${all_healthy}" = true ]; then
        log_info "All health checks passed! ✓"
        exit 0
    else
        log_error "Some health checks failed ✗"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ip)
            INSTANCE_IP="$2"
            shift 2
            ;;
        --port)
            APP_PORT="$2"
            shift 2
            ;;
        --key)
            AWS_KEY_PATH="$2"
            shift 2
            ;;
        --docker)
            CHECK_DOCKER=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main

