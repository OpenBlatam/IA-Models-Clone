#!/bin/bash
# Monitoring script for 3D Prototype AI deployment
# Provides real-time monitoring and metrics

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Default values
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"
readonly MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Monitor 3D Prototype AI deployment.

OPTIONS:
    -i, --ip IP              Instance IP address (required)
    -k, --key-path PATH     Path to SSH private key
    -n, --interval SECONDS   Monitoring interval in seconds (default: 5)
    -h, --help              Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem
    $0 --ip 1.2.3.4 --interval 10

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
            -n|--interval)
                MONITOR_INTERVAL="$2"
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

# Get application metrics
get_app_metrics() {
    local ip="${1}"
    
    local health_status
    health_status=$(curl -sf "http://${ip}/health" 2>/dev/null && echo "✓" || echo "✗")
    
    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://${ip}/health" 2>/dev/null || echo "N/A")
    
    echo "${health_status}|${response_time}"
}

# Get system metrics via SSH
get_system_metrics() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${key_path}" ]; then
        echo "N/A|N/A|N/A|N/A"
        return 0
    fi
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=2 \
        ubuntu@${ip} \
        "echo \$(free -h | awk '/^Mem:/ {print \$3\"/\"\$2}')|\$(df -h / | tail -1 | awk '{print \$5}')|\$(uptime | awk -F'load average:' '{print \$2}')|\$(docker ps --format '{{.Names}}' | wc -l)" 2>/dev/null || echo "N/A|N/A|N/A|N/A"
}

# Display monitoring dashboard
display_dashboard() {
    local ip="${1}"
    local key_path="${2}"
    
    clear
    echo "=========================================="
    echo "3D Prototype AI - Monitoring Dashboard"
    echo "=========================================="
    echo "Instance: ${ip}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Get metrics
    local app_metrics
    app_metrics=$(get_app_metrics "${ip}")
    local health_status
    health_status=$(echo "${app_metrics}" | cut -d'|' -f1)
    local response_time
    response_time=$(echo "${app_metrics}" | cut -d'|' -f2)
    
    local system_metrics
    system_metrics=$(get_system_metrics "${ip}" "${key_path}")
    local memory
    memory=$(echo "${system_metrics}" | cut -d'|' -f1)
    local disk
    disk=$(echo "${system_metrics}" | cut -d'|' -f2)
    local load
    load=$(echo "${system_metrics}" | cut -d'|' -f3)
    local containers
    containers=$(echo "${system_metrics}" | cut -d'|' -f4)
    
    # Display metrics
    printf "Application Health: %s\n" "${health_status}"
    printf "Response Time:     %s s\n" "${response_time}"
    echo ""
    printf "Memory Usage:      %s\n" "${memory}"
    printf "Disk Usage:       %s\n" "${disk}"
    printf "Load Average:     %s\n" "${load}"
    printf "Running Containers: %s\n" "${containers}"
    echo ""
    echo "=========================================="
    echo "Press Ctrl+C to exit"
    echo "=========================================="
}

# Main monitoring loop
main() {
    parse_args "$@"
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "Instance IP is required. Use --ip option"
    fi
    
    validate_ip "${INSTANCE_IP}"
    
    log_info "Starting monitoring for instance: ${INSTANCE_IP}"
    log_info "Monitoring interval: ${MONITOR_INTERVAL} seconds"
    log_info "Press Ctrl+C to stop"
    echo ""
    
    # Monitoring loop
    while true; do
        display_dashboard "${INSTANCE_IP}" "${AWS_KEY_PATH}"
        sleep "${MONITOR_INTERVAL}"
    done
}

# Handle Ctrl+C
trap 'log_info "Monitoring stopped"; exit 0' INT TERM

main "$@"

