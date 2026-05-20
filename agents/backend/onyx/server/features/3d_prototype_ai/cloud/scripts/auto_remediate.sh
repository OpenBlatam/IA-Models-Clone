#!/bin/bash
# Automated remediation script
# Automatically fixes common issues

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
readonly AUTO_FIX="${AUTO_FIX:-false}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Automated remediation for common issues.

COMMANDS:
    check                Check for issues
    fix                  Fix detected issues
    monitor              Continuous monitoring and auto-fix
    status               Show remediation status

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -a, --auto-fix           Automatically fix issues
    -h, --help               Show this help message

EXAMPLES:
    $0 check --ip 1.2.3.4
    $0 fix --ip 1.2.3.4 --auto-fix
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
            -a|--auto-fix)
                AUTO_FIX=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            check|fix|monitor|status)
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

# Check for issues
check_issues() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Checking for issues..."
    
    local issues=()
    
    # Check disk space
    local disk_usage
    disk_usage=$(ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} \
        "df -h / | tail -1 | awk '{print \$5}' | sed 's/%//'" 2>/dev/null || echo "0")
    
    if [ "${disk_usage}" -gt 90 ]; then
        issues+=("High disk usage: ${disk_usage}%")
    fi
    
    # Check memory
    local mem_usage
    mem_usage=$(ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} \
        "free | awk '/^Mem:/ {printf \"%.0f\", \$3/\$2 * 100}'" 2>/dev/null || echo "0")
    
    if [ "${mem_usage}" -gt 90 ]; then
        issues+=("High memory usage: ${mem_usage}%")
    fi
    
    # Check application health
    if ! curl -sf "http://${ip}:8030/health" > /dev/null 2>&1; then
        issues+=("Application unhealthy")
    fi
    
    # Report issues
    if [ ${#issues[@]} -eq 0 ]; then
        log_info "✓ No issues detected"
        return 0
    else
        log_warn "Issues detected:"
        for issue in "${issues[@]}"; do
            log_warn "  - ${issue}"
        done
        return 1
    fi
}

# Fix issues
fix_issues() {
    local ip="${1}"
    local key_path="${2}"
    local auto_fix="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    if [ "${auto_fix}" != "true" ]; then
        log_warn "This will attempt to fix detected issues"
        read -p "Continue? (yes/no): " confirm
        if [ "${confirm}" != "yes" ]; then
            log_info "Fix cancelled"
            return 0
        fi
    fi
    
    log_info "Fixing issues..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e

# Fix disk space
DISK_USAGE=\$(df -h / | tail -1 | awk '{print \$5}' | sed 's/%//')
if [ \$DISK_USAGE -gt 90 ]; then
    echo "Fixing disk space..."
    # Clean Docker
    docker system prune -af --volumes || true
    # Clean logs
    sudo journalctl --vacuum-time=7d || true
    # Clean temp files
    sudo find /tmp -type f -mtime +7 -delete || true
    echo "✓ Disk space cleaned"
fi

# Fix memory
MEM_USAGE=\$(free | awk '/^Mem:/ {printf "%.0f", \$3/\$2 * 100}')
if [ \$MEM_USAGE -gt 90 ]; then
    echo "Fixing memory usage..."
    # Restart services
    if [ -f "/opt/3d-prototype-ai/docker-compose.yml" ]; then
        cd /opt/3d-prototype-ai
        docker-compose restart || true
    fi
    echo "✓ Memory usage addressed"
fi

# Fix application
if ! curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "Fixing application..."
    if [ -f "/opt/3d-prototype-ai/docker-compose.yml" ]; then
        cd /opt/3d-prototype-ai
        docker-compose restart api || docker-compose up -d
    else
        sudo systemctl restart 3d-prototype-ai || true
    fi
    echo "✓ Application restarted"
fi

echo "✅ Remediation completed"
REMOTE_EOF
    
    log_info "Issues fixed"
}

# Monitor and auto-fix
monitor_and_fix() {
    local ip="${1}"
    local key_path="${2}"
    local interval="${3:-300}"
    
    log_info "Starting automated remediation monitor (interval: ${interval}s)..."
    
    while true; do
        if check_issues "${ip}" "${key_path}"; then
            log_info "No issues detected, continuing monitoring..."
        else
            log_warn "Issues detected, attempting auto-fix..."
            fix_issues "${ip}" "${key_path}" "true"
        fi
        
        sleep "${interval}"
    done
}

# Show status
show_status() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Remediation status:"
    
    check_issues "${ip}" "${key_path}"
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
        check)
            check_issues "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        fix)
            fix_issues "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${AUTO_FIX}"
            ;;
        monitor)
            monitor_and_fix "${INSTANCE_IP}" "${AWS_KEY_PATH}" 300
            ;;
        status)
            show_status "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


