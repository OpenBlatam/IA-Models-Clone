#!/bin/bash
# Canary deployment script
# Implements canary deployment strategy

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
readonly CANARY_PERCENTAGE="${CANARY_PERCENTAGE:-10}"
readonly CANARY_DURATION="${CANARY_DURATION:-300}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Canary deployment management.

COMMANDS:
    deploy              Deploy canary version
    promote             Promote canary to full deployment
    rollback            Rollback canary deployment
    status              Show canary status
    monitor             Monitor canary metrics

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -p, --percentage PERCENT Canary traffic percentage (default: 10)
    -d, --duration SECONDS   Canary duration in seconds (default: 300)
    -h, --help               Show this help message

EXAMPLES:
    $0 deploy --ip 1.2.3.4 --percentage 10
    $0 promote --ip 1.2.3.4
    $0 monitor --ip 1.2.3.4

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
            -p|--percentage)
                CANARY_PERCENTAGE="$2"
                shift 2
                ;;
            -d|--duration)
                CANARY_DURATION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            deploy|promote|rollback|status|monitor)
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

# Deploy canary
deploy_canary() {
    local ip="${1}"
    local key_path="${2}"
    local percentage="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Deploying canary version (${percentage}% traffic)..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Create canary environment
mkdir -p canary
cp -r . canary/ 2>/dev/null || true

# Deploy canary version
cd canary
if [ -f "docker-compose.yml" ]; then
    # Update canary configuration
    sed -i 's/8030/8032/g' docker-compose.yml 2>/dev/null || true
    
    # Deploy canary
    docker-compose -p canary up -d --build
    sleep 10
    
    # Health check
    if curl -f http://localhost:8032/health > /dev/null 2>&1; then
        echo "✅ Canary deployed successfully"
    else
        echo "❌ Canary health check failed"
        exit 1
    fi
fi

# Configure load balancer for canary traffic
# This is a placeholder - implement based on your load balancer
echo "Canary traffic: ${percentage}%"
REMOTE_EOF
    
    log_info "Canary deployment completed (${percentage}% traffic)"
}

# Promote canary
promote_canary() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "Promoting canary to full deployment..."
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Promotion cancelled"
        return 0
    fi
    
    log_info "Promoting canary..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Switch all traffic to canary
if [ -f "docker-compose.yml" ]; then
    # Stop old version
    docker-compose stop api || true
    
    # Promote canary
    docker-compose -p canary up -d
    
    # Update main deployment
    docker-compose up -d
    
    echo "✅ Canary promoted to full deployment"
fi
REMOTE_EOF
    
    log_info "Canary promoted successfully"
}

# Rollback canary
rollback_canary() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "Rolling back canary deployment..."
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Rollback cancelled"
        return 0
    fi
    
    log_info "Rolling back canary..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e
cd /opt/3d-prototype-ai

# Stop canary
docker-compose -p canary down || true

# Restore original
docker-compose up -d

echo "✅ Canary rolled back"
REMOTE_EOF
    
    log_info "Canary rolled back successfully"
}

# Show status
show_status() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Canary deployment status:"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
cd /opt/3d-prototype-ai

echo "Production (port 8030):"
if curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "  ✅ Healthy"
else
    echo "  ❌ Unhealthy"
fi

echo ""
echo "Canary (port 8032):"
if curl -f http://localhost:8032/health > /dev/null 2>&1; then
    echo "  ✅ Healthy"
    echo "  Traffic: ${CANARY_PERCENTAGE}%"
else
    echo "  ❌ Not deployed or unhealthy"
fi

echo ""
echo "Containers:"
docker-compose ps 2>/dev/null || echo "  No production containers"
docker-compose -p canary ps 2>/dev/null || echo "  No canary containers"
REMOTE_EOF
}

# Monitor canary
monitor_canary() {
    local ip="${1}"
    local key_path="${2}"
    local duration="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Monitoring canary for ${duration} seconds..."
    
    local end_time
    end_time=$(($(date +%s) + duration))
    local check_count=0
    local success_count=0
    local fail_count=0
    
    while [ $(date +%s) -lt $end_time ]; do
        check_count=$((check_count + 1))
        
        # Check canary health
        if ssh -i "${key_path}" \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            ubuntu@${ip} \
            "curl -f http://localhost:8032/health > /dev/null 2>&1" 2>/dev/null; then
            success_count=$((success_count + 1))
            printf "[%s] ✓ Canary healthy (Success: %d/%d)\n" \
                "$(date '+%H:%M:%S')" "${success_count}" "${check_count}"
        else
            fail_count=$((fail_count + 1))
            printf "[%s] ✗ Canary unhealthy (Fails: %d/%d)\n" \
                "$(date '+%H:%M:%S')" "${fail_count}" "${check_count}"
        fi
        
        sleep 10
    done
    
    local success_rate
    success_rate=$(echo "scale=2; (${success_count} * 100) / ${check_count}" | bc)
    
    echo ""
    log_info "Monitoring Summary:"
    log_info "  Total checks: ${check_count}"
    log_info "  Successful: ${success_count}"
    log_info "  Failed: ${fail_count}"
    log_info "  Success rate: ${success_rate}%"
    
    if (( $(echo "${success_rate} > 95" | bc -l) )); then
        log_info "✅ Canary is performing well"
    else
        log_warn "⚠️ Canary has issues - consider rollback"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        if [ "${COMMAND}" != "status" ]; then
            error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
        fi
    fi
    
    case "${COMMAND}" in
        deploy)
            deploy_canary "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${CANARY_PERCENTAGE}"
            ;;
        promote)
            promote_canary "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        rollback)
            rollback_canary "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        status)
            show_status "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        monitor)
            monitor_canary "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${CANARY_DURATION}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


