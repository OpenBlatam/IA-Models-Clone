#!/bin/bash
# Health monitoring script with alerting
# Continuously monitors application health and sends alerts

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
readonly CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
readonly MAX_FAILURES="${MAX_FAILURES:-3}"
readonly ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Monitor application health continuously.

OPTIONS:
    -i, --ip IP              Instance IP address (required)
    -k, --key-path PATH      Path to SSH private key
    -c, --interval SECONDS   Check interval in seconds (default: 30)
    -m, --max-failures NUM   Max failures before alert (default: 3)
    -w, --webhook URL        Webhook URL for alerts
    -d, --daemon             Run as daemon
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --interval 60
    $0 --ip 1.2.3.4 --daemon --webhook https://hooks.slack.com/...

EOF
}

# Parse arguments
parse_args() {
    DAEMON=false
    
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
            -c|--interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            -m|--max-failures)
                MAX_FAILURES="$2"
                shift 2
                ;;
            -w|--webhook)
                ALERT_WEBHOOK="$2"
                shift 2
                ;;
            -d|--daemon)
                DAEMON=true
                shift
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

# Check application health
check_health() {
    local ip="${1}"
    
    local response
    response=$(curl -sf -m 10 "http://${ip}:8030/health" 2>/dev/null)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ] && [ -n "${response}" ]; then
        echo "healthy|${response}"
    else
        echo "unhealthy|"
    fi
}

# Check system resources
check_resources() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${key_path}" ]; then
        echo "N/A|N/A|N/A"
        return 0
    fi
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 \
        ubuntu@${ip} << 'REMOTE_EOF' 2>/dev/null || echo "N/A|N/A|N/A"
CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.1f", 100 - $1}')
MEM=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
DISK=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
echo "${CPU}|${MEM}|${DISK}"
REMOTE_EOF
}

# Send alert
send_alert() {
    local message="${1}"
    local severity="${2:-warning}"
    
    log_error "ALERT [${severity}]: ${message}"
    
    if [ -n "${ALERT_WEBHOOK}" ]; then
        local color
        case "${severity}" in
            critical)
                color="danger"
                emoji="🚨"
                ;;
            warning)
                color="warning"
                emoji="⚠️"
                ;;
            *)
                color="good"
                emoji="ℹ️"
                ;;
        esac
        
        local payload
        payload=$(cat << EOF
{
  "text": "${emoji} Health Monitor Alert",
  "attachments": [
    {
      "color": "${color}",
      "text": "${message}",
      "fields": [
        {
          "title": "Time",
          "value": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
          "short": true
        },
        {
          "title": "Severity",
          "value": "${severity}",
          "short": true
        }
      ]
    }
  ]
}
EOF
)
        
        curl -X POST -H 'Content-type: application/json' \
            --data "${payload}" \
            "${ALERT_WEBHOOK}" 2>/dev/null || true
    fi
}

# Monitor loop
monitor_loop() {
    local ip="${1}"
    local key_path="${2}"
    local interval="${3}"
    local max_failures="${4}"
    
    local failure_count=0
    local consecutive_failures=0
    
    log_info "Starting health monitor for ${ip} (interval: ${interval}s)"
    
    while true; do
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Check health
        local health_result
        health_result=$(check_health "${ip}")
        local health_status
        health_status=$(echo "${health_result}" | cut -d'|' -f1)
        local health_data
        health_data=$(echo "${health_result}" | cut -d'|' -f2)
        
        # Check resources
        local resources
        resources=$(check_resources "${ip}" "${key_path}")
        local cpu
        cpu=$(echo "${resources}" | cut -d'|' -f1)
        local mem
        mem=$(echo "${resources}" | cut -d'|' -f2)
        local disk
        disk=$(echo "${resources}" | cut -d'|' -f3)
        
        # Evaluate health
        if [ "${health_status}" = "healthy" ]; then
            if [ $consecutive_failures -gt 0 ]; then
                log_info "Application recovered after ${consecutive_failures} failures"
                send_alert "Application recovered: ${ip}" "info"
            fi
            consecutive_failures=0
            
            # Check resource thresholds
            if [ "${cpu}" != "N/A" ]; then
                local cpu_float
                cpu_float=$(echo "${cpu}" | awk '{print int($1)}')
                if [ $cpu_float -gt 90 ]; then
                    send_alert "High CPU usage: ${cpu}% on ${ip}" "warning"
                fi
            fi
            
            if [ "${mem}" != "N/A" ]; then
                local mem_float
                mem_float=$(echo "${mem}" | awk '{print int($1)}')
                if [ $mem_float -gt 90 ]; then
                    send_alert "High memory usage: ${mem}% on ${ip}" "warning"
                fi
            fi
            
            if [ "${disk}" != "N/A" ]; then
                local disk_int
                disk_int=$(echo "${disk}" | awk '{print int($1)}')
                if [ $disk_int -gt 90 ]; then
                    send_alert "High disk usage: ${disk}% on ${ip}" "critical"
                fi
            fi
            
            printf "[%s] ✓ Healthy | CPU: %s%% | Mem: %s%% | Disk: %s%%\n" \
                "${timestamp}" "${cpu:-N/A}" "${mem:-N/A}" "${disk:-N/A}"
        else
            consecutive_failures=$((consecutive_failures + 1))
            failure_count=$((failure_count + 1))
            
            printf "[%s] ✗ Unhealthy (failures: %d)\n" "${timestamp}" "${consecutive_failures}"
            
            if [ $consecutive_failures -ge $max_failures ]; then
                send_alert "Application unhealthy: ${ip} (${consecutive_failures} consecutive failures)" "critical"
            fi
        fi
        
        sleep "${interval}"
    done
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "INSTANCE_IP is required"
    fi
    
    validate_ip "${INSTANCE_IP}"
    
    if [ "${DAEMON}" = "true" ]; then
        log_info "Running as daemon..."
        monitor_loop "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${CHECK_INTERVAL}" "${MAX_FAILURES}" &
        echo $! > /tmp/health_monitor_${INSTANCE_IP//./_}.pid
        log_info "Daemon started (PID: $(cat /tmp/health_monitor_${INSTANCE_IP//./_}.pid))"
    else
        monitor_loop "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${CHECK_INTERVAL}" "${MAX_FAILURES}"
    fi
}

main "$@"


