#!/bin/bash
# Optimization script
# Optimizes system and application performance

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

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Optimize system and application performance.

COMMANDS:
    system              Optimize system settings
    application         Optimize application
    database            Optimize database (if applicable)
    network             Optimize network settings
    docker              Optimize Docker configuration
    all                 Run all optimizations

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -h, --help               Show this help message

EXAMPLES:
    $0 system --ip 1.2.3.4
    $0 all --ip 1.2.3.4

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
            system|application|database|network|docker|all)
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

# Optimize system
optimize_system() {
    local ip="${1}"
    local key_path="${2}"
    
    log_info "Optimizing system settings..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
set -e

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install optimization tools
sudo apt-get install -y htop iotop nethogs

# Optimize swappiness
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl -w vm.swappiness=10

# Optimize file handles
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
cat << SYSCTL | sudo tee -a /etc/sysctl.conf
# Network optimizations
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.ipv4.tcp_congestion_control = bbr
SYSCTL

sudo sysctl -p

echo "✅ System optimization completed"
REMOTE_EOF
}

# Optimize application
optimize_application() {
    local ip="${1}"
    local key_path="${2}"
    
    log_info "Optimizing application..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
set -e
cd /opt/3d-prototype-ai

# Clean Python cache
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Optimize Docker if using
if [ -f "docker-compose.yml" ]; then
    # Prune unused resources
    docker system prune -f
    
    # Optimize images
    docker images | grep -v REPOSITORY | awk '{print $3}' | xargs docker rmi 2>/dev/null || true
fi

echo "✅ Application optimization completed"
REMOTE_EOF
}

# Optimize Docker
optimize_docker() {
    local ip="${1}"
    local key_path="${2}"
    
    log_info "Optimizing Docker..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
set -e

# Prune system
docker system prune -af --volumes

# Clean build cache
docker builder prune -af

# Optimize Docker daemon
sudo tee /etc/docker/daemon.json > /dev/null << DOCKER_CONFIG
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
DOCKER_CONFIG

sudo systemctl restart docker

echo "✅ Docker optimization completed"
REMOTE_EOF
}

# Optimize network
optimize_network() {
    local ip="${1}"
    local key_path="${2}"
    
    log_info "Optimizing network settings..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
set -e

# Network optimizations
cat << NETWORK | sudo tee -a /etc/sysctl.conf
# Network performance
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_tw_reuse = 1
net.ipv4.ip_local_port_range = 1024 65535
NETWORK

sudo sysctl -p

echo "✅ Network optimization completed"
REMOTE_EOF
}

# Run all optimizations
optimize_all() {
    local ip="${1}"
    local key_path="${2}"
    
    log_info "Running all optimizations..."
    
    optimize_system "${ip}" "${key_path}"
    optimize_application "${ip}" "${key_path}"
    optimize_docker "${ip}" "${key_path}"
    optimize_network "${ip}" "${key_path}"
    
    log_info "All optimizations completed!"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    validate_ip "${INSTANCE_IP}"
    validate_file "${AWS_KEY_PATH}" "SSH private key"
    
    case "${COMMAND}" in
        system)
            optimize_system "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        application)
            optimize_application "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        database)
            log_info "Database optimization - implement based on your database"
            ;;
        network)
            optimize_network "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        docker)
            optimize_docker "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        all)
            optimize_all "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


