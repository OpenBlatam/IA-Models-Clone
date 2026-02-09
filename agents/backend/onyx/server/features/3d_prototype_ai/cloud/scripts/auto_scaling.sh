#!/bin/bash
# Auto-scaling script
# Automatically scales resources based on metrics

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
readonly CPU_THRESHOLD_HIGH="${CPU_THRESHOLD_HIGH:-80}"
readonly CPU_THRESHOLD_LOW="${CPU_THRESHOLD_LOW:-20}"
readonly MEMORY_THRESHOLD_HIGH="${MEMORY_THRESHOLD_HIGH:-85}"
readonly MEMORY_THRESHOLD_LOW="${MEMORY_THRESHOLD_LOW:-30}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Auto-scaling management.

COMMANDS:
    check                Check if scaling is needed
    scale-up             Scale up resources
    scale-down           Scale down resources
    monitor              Continuous monitoring and auto-scaling
    status               Show scaling status

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -c, --cpu-high PERCENT   CPU threshold for scale-up (default: 80)
    -m, --memory-high PERCENT Memory threshold for scale-up (default: 85)
    -h, --help               Show this help message

EXAMPLES:
    $0 check --ip 1.2.3.4
    $0 monitor --ip 1.2.3.4 --cpu-high 75

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
            -c|--cpu-high)
                CPU_THRESHOLD_HIGH="$2"
                shift 2
                ;;
            -m|--memory-high)
                MEMORY_THRESHOLD_HIGH="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            check|scale-up|scale-down|monitor|status)
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

# Get current metrics
get_metrics() {
    local ip="${1}"
    local key_path="${2}"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF' 2>/dev/null || echo "N/A|N/A|N/A"
CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.1f", 100 - $1}')
MEM=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
echo "${CPU}|${MEM}|${LOAD}"
REMOTE_EOF
}

# Check if scaling is needed
check_scaling() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Checking if scaling is needed..."
    
    local metrics
    metrics=$(get_metrics "${ip}" "${key_path}")
    local cpu
    cpu=$(echo "${metrics}" | cut -d'|' -f1)
    local mem
    mem=$(echo "${metrics}" | cut -d'|' -f2)
    local load
    load=$(echo "${metrics}" | cut -d'|' -f3)
    
    echo "Current Metrics:"
    echo "  CPU: ${cpu}%"
    echo "  Memory: ${mem}%"
    echo "  Load: ${load}"
    echo ""
    
    local cpu_float
    cpu_float=$(echo "${cpu}" | awk '{print int($1)}')
    local mem_float
    mem_float=$(echo "${mem}" | awk '{print int($1)}')
    
    if [ "${cpu_float}" -gt "${CPU_THRESHOLD_HIGH}" ] || [ "${mem_float}" -gt "${MEMORY_THRESHOLD_HIGH}" ]; then
        echo "⚠️ Scale-up recommended"
        echo "  CPU threshold: ${CPU_THRESHOLD_HIGH}%"
        echo "  Memory threshold: ${MEMORY_THRESHOLD_HIGH}%"
        return 1
    elif [ "${cpu_float}" -lt "${CPU_THRESHOLD_LOW}" ] && [ "${mem_float}" -lt "${MEMORY_THRESHOLD_LOW}" ]; then
        echo "ℹ️ Scale-down possible"
        echo "  CPU threshold: ${CPU_THRESHOLD_LOW}%"
        echo "  Memory threshold: ${MEMORY_THRESHOLD_LOW}%"
        return 2
    else
        echo "✅ No scaling needed"
        return 0
    fi
}

# Scale up
scale_up() {
    local ip="${1}"
    local key_path="${2}"
    
    log_info "Scaling up resources..."
    
    # For Docker, scale containers
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
cd /opt/3d-prototype-ai

if [ -f "docker-compose.yml" ]; then
    # Scale API containers
    docker-compose up -d --scale api=2 2>/dev/null || \
    echo "⚠️ Could not scale containers (may need to update docker-compose.yml)"
    
    echo "✅ Scaling initiated"
else
    echo "ℹ️ Docker Compose not available, manual scaling required"
fi
REMOTE_EOF
}

# Scale down
scale_down() {
    local ip="${1}"
    local key_path="${2}"
    
    log_info "Scaling down resources..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
cd /opt/3d-prototype-ai

if [ -f "docker-compose.yml" ]; then
    # Scale back to 1
    docker-compose up -d --scale api=1 2>/dev/null || \
    echo "⚠️ Could not scale down"
    
    echo "✅ Scaling down completed"
else
    echo "ℹ️ Docker Compose not available, manual scaling required"
fi
REMOTE_EOF
}

# Monitor and auto-scale
monitor_auto_scale() {
    local ip="${1}"
    local key_path="${2}"
    local interval="${3:-60}"
    
    log_info "Starting auto-scaling monitor (interval: ${interval}s)..."
    
    while true; do
        local metrics
        metrics=$(get_metrics "${ip}" "${key_path}")
        local cpu
        cpu=$(echo "${metrics}" | cut -d'|' -f1)
        local mem
        mem=$(echo "${metrics}" | cut -d'|' -f2)
        
        local cpu_float
        cpu_float=$(echo "${cpu}" | awk '{print int($1)}')
        local mem_float
        mem_float=$(echo "${mem}" | awk '{print int($1)}')
        
        if [ "${cpu_float}" -gt "${CPU_THRESHOLD_HIGH}" ] || [ "${mem_float}" -gt "${MEMORY_THRESHOLD_HIGH}" ]; then
            log_warn "High resource usage detected (CPU: ${cpu}%, Mem: ${mem}%)"
            scale_up "${ip}" "${key_path}"
        elif [ "${cpu_float}" -lt "${CPU_THRESHOLD_LOW}" ] && [ "${mem_float}" -lt "${MEMORY_THRESHOLD_LOW}" ]; then
            log_info "Low resource usage detected (CPU: ${cpu}%, Mem: ${mem}%)"
            # Only scale down if consistently low
            scale_down "${ip}" "${key_path}"
        fi
        
        sleep "${interval}"
    done
}

# Show scaling status
show_status() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Scaling status..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF'
echo "Auto-Scaling Status"
echo "=================="
echo ""

# Get metrics
CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.1f", 100 - $1}')
MEM=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')

echo "Current Metrics:"
echo "  CPU Usage: ${CPU}%"
echo "  Memory Usage: ${MEM}%"
echo "  Load Average: ${LOAD}"
echo ""

# Check Docker scaling
if [ -f "/opt/3d-prototype-ai/docker-compose.yml" ]; then
    echo "Docker Containers:"
    cd /opt/3d-prototype-ai
    docker-compose ps 2>/dev/null || echo "  Could not get container status"
else
    echo "Docker Compose: Not configured"
fi
REMOTE_EOF
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        check)
            check_scaling "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        scale-up)
            scale_up "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        scale-down)
            scale_down "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        monitor)
            monitor_auto_scale "${INSTANCE_IP}" "${AWS_KEY_PATH}" 60
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


