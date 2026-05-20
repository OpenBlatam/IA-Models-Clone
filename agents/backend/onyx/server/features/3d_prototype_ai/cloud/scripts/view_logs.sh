#!/bin/bash
# View application logs from EC2 instance

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
AWS_KEY_PATH="${AWS_KEY_PATH}"
LOG_TYPE="${1:-app}"
LINES="${2:-50}"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Check prerequisites
if [ -z "${INSTANCE_IP}" ]; then
    echo "Usage: $0 [app|nginx|system|docker] [lines]"
    echo "Or set INSTANCE_IP in .env file"
    exit 1
fi

if [ -z "${AWS_KEY_PATH}" ]; then
    echo "AWS_KEY_PATH not set in .env file"
    exit 1
fi

case "${LOG_TYPE}" in
    app)
        log_info "Viewing application logs..."
        ssh -i "${AWS_KEY_PATH}" \
            -o StrictHostKeyChecking=no \
            ubuntu@${INSTANCE_IP} << EOF
cd /opt/3d-prototype-ai
if [ -f "docker-compose.yml" ]; then
    docker-compose logs --tail=${LINES} -f
else
    sudo journalctl -u 3d-prototype-ai -n ${LINES} -f
fi
EOF
        ;;
    nginx)
        log_info "Viewing Nginx logs..."
        ssh -i "${AWS_KEY_PATH}" \
            -o StrictHostKeyChecking=no \
            ubuntu@${INSTANCE_IP} "sudo tail -f /var/log/nginx/access.log /var/log/nginx/error.log"
        ;;
    system)
        log_info "Viewing system logs..."
        ssh -i "${AWS_KEY_PATH}" \
            -o StrictHostKeyChecking=no \
            ubuntu@${INSTANCE_IP} "sudo journalctl -n ${LINES} -f"
        ;;
    docker)
        log_info "Viewing Docker logs..."
        ssh -i "${AWS_KEY_PATH}" \
            -o StrictHostKeyChecking=no \
            ubuntu@${INSTANCE_IP} << EOF
cd /opt/3d-prototype-ai
docker-compose logs --tail=${LINES} -f
EOF
        ;;
    *)
        echo "Unknown log type: ${LOG_TYPE}"
        echo "Available types: app, nginx, system, docker"
        exit 1
        ;;
esac

