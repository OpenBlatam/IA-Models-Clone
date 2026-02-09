#!/bin/bash
# Chaos engineering script
# Implements chaos engineering practices

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
readonly CHAOS_TYPE="${CHAOS_TYPE:-cpu}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Chaos engineering experiments.

COMMANDS:
    inject              Inject chaos
    stop                Stop chaos
    status              Show chaos status
    experiment          Run chaos experiment

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -t, --type TYPE          Chaos type (cpu|memory|network|disk) (default: cpu)
    -d, --duration SECONDS   Chaos duration (default: 60)
    -h, --help               Show this help message

EXAMPLES:
    $0 inject --type cpu --duration 120
    $0 experiment --type network
    $0 stop

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    DURATION=60
    
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
            -t|--type)
                CHAOS_TYPE="$2"
                shift 2
                ;;
            -d|--duration)
                DURATION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            inject|stop|status|experiment)
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

# Inject chaos
inject_chaos() {
    local ip="${1}"
    local key_path="${2}"
    local chaos_type="${3}"
    local duration="${4}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "Injecting chaos: ${chaos_type} for ${duration} seconds"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Chaos injection cancelled"
        return 0
    fi
    
    log_info "Injecting ${chaos_type} chaos..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e

case "${chaos_type}" in
    cpu)
        echo "Injecting CPU stress..."
        stress-ng --cpu 4 --timeout ${duration}s &
        echo \$! > /tmp/chaos.pid
        ;;
    memory)
        echo "Injecting memory stress..."
        stress-ng --vm 2 --vm-bytes 1G --timeout ${duration}s &
        echo \$! > /tmp/chaos.pid
        ;;
    network)
        echo "Injecting network chaos..."
        # Simulate network issues
        sudo tc qdisc add dev eth0 root netem delay 100ms loss 1% &
        echo \$! > /tmp/chaos.pid
        ;;
    disk)
        echo "Injecting disk I/O stress..."
        stress-ng --io 4 --timeout ${duration}s &
        echo \$! > /tmp/chaos.pid
        ;;
esac

echo "Chaos injected (PID: \$(cat /tmp/chaos.pid))"
REMOTE_EOF
    
    log_info "Chaos injected for ${duration} seconds"
}

# Stop chaos
stop_chaos() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Stopping chaos..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
if [ -f /tmp/chaos.pid ]; then
    PID=\$(cat /tmp/chaos.pid)
    kill \${PID} 2>/dev/null || true
    rm /tmp/chaos.pid
    echo "Chaos stopped"
else
    echo "No active chaos"
fi

# Cleanup network chaos
sudo tc qdisc del dev eth0 root 2>/dev/null || true
REMOTE_EOF
    
    log_info "Chaos stopped"
}

# Show status
show_status() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Chaos engineering status:"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
if [ -f /tmp/chaos.pid ]; then
    PID=\$(cat /tmp/chaos.pid)
    if kill -0 \${PID} 2>/dev/null; then
        echo "Chaos active (PID: \${PID})"
    else
        echo "Chaos not running"
    fi
else
    echo "No chaos active"
fi
REMOTE_EOF
}

# Run experiment
run_experiment() {
    local ip="${1}"
    local key_path="${2}"
    local chaos_type="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Running chaos experiment: ${chaos_type}"
    
    # Baseline metrics
    log_info "Collecting baseline metrics..."
    ./scripts/metrics.sh --ip "${ip}" --key-path "${key_path}" --json > /tmp/baseline.json
    
    # Inject chaos
    inject_chaos "${ip}" "${key_path}" "${chaos_type}" 120
    
    # Monitor during chaos
    log_info "Monitoring during chaos..."
    sleep 30
    
    # Collect metrics during chaos
    ./scripts/metrics.sh --ip "${ip}" --key-path "${key_path}" --json > /tmp/chaos_metrics.json
    
    # Stop chaos
    stop_chaos "${ip}" "${key_path}"
    
    # Recovery metrics
    log_info "Monitoring recovery..."
    sleep 30
    ./scripts/metrics.sh --ip "${ip}" --key-path "${key_path}" --json > /tmp/recovery.json
    
    # Generate report
    log_info "Chaos experiment completed"
    log_info "Baseline: /tmp/baseline.json"
    log_info "During chaos: /tmp/chaos_metrics.json"
    log_info "Recovery: /tmp/recovery.json"
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
        inject)
            inject_chaos "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${CHAOS_TYPE}" "${DURATION}"
            ;;
        stop)
            stop_chaos "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        status)
            show_status "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        experiment)
            run_experiment "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${CHAOS_TYPE}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


