#!/bin/bash
# Advanced monitoring script
# Provides comprehensive monitoring and alerting

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
readonly MONITORING_INTERVAL="${MONITORING_INTERVAL:-60}"
readonly METRICS_RETENTION="${METRICS_RETENTION:-30}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Advanced monitoring and alerting.

COMMANDS:
    start               Start monitoring
    stop                Stop monitoring
    status              Show monitoring status
    metrics             Collect metrics
    alerts              Show active alerts
    dashboard           Generate monitoring dashboard

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -t, --interval SECONDS    Monitoring interval (default: 60)
    -h, --help               Show this help message

EXAMPLES:
    $0 start --ip 1.2.3.4
    $0 metrics --ip 1.2.3.4
    $0 dashboard

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
            -t|--interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            start|stop|status|metrics|alerts|dashboard)
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

# Start monitoring
start_monitoring() {
    local ip="${1}"
    local key_path="${2}"
    local interval="${3}"
    
    log_info "Starting advanced monitoring..."
    
    # Start monitoring daemon
    nohup "${SCRIPT_DIR}/health_monitor.sh" \
        --ip "${ip}" \
        --key-path "${key_path}" \
        --interval "${interval}" \
        --daemon \
        > /tmp/monitoring.log 2>&1 &
    
    echo $! > /tmp/monitoring.pid
    
    log_info "Monitoring started (PID: $(cat /tmp/monitoring.pid))"
}

# Stop monitoring
stop_monitoring() {
    if [ -f /tmp/monitoring.pid ]; then
        local pid
        pid=$(cat /tmp/monitoring.pid)
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
            rm /tmp/monitoring.pid
            log_info "Monitoring stopped"
        else
            log_warn "Monitoring not running"
            rm /tmp/monitoring.pid
        fi
    else
        log_warn "Monitoring PID file not found"
    fi
}

# Show status
show_status() {
    if [ -f /tmp/monitoring.pid ]; then
        local pid
        pid=$(cat /tmp/monitoring.pid)
        if kill -0 "${pid}" 2>/dev/null; then
            log_info "Monitoring is running (PID: ${pid})"
        else
            log_warn "Monitoring is not running"
        fi
    else
        log_warn "Monitoring is not running"
    fi
}

# Collect metrics
collect_metrics() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Collecting comprehensive metrics..."
    
    ./scripts/metrics.sh --ip "${ip}" --key-path "${key_path}" --json
}

# Show alerts
show_alerts() {
    log_info "Active alerts:"
    
    # Check alert log
    if [ -f /tmp/alerts.log ]; then
        tail -20 /tmp/alerts.log
    else
        log_info "No active alerts"
    fi
}

# Generate dashboard
generate_dashboard() {
    local ip="${1}"
    
    log_info "Generating monitoring dashboard..."
    
    # Use dashboard script
    if [ -n "${ip}" ]; then
        ./scripts/dashboard.sh --ip "${ip}" --refresh 30
    else
        log_info "Monitoring dashboard - use dashboard.sh for interactive dashboard"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        start)
            if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
                error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
            fi
            start_monitoring "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${MONITORING_INTERVAL}"
            ;;
        stop)
            stop_monitoring
            ;;
        status)
            show_status
            ;;
        metrics)
            if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
                error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
            fi
            collect_metrics "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        alerts)
            show_alerts
            ;;
        dashboard)
            generate_dashboard "${INSTANCE_IP}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


