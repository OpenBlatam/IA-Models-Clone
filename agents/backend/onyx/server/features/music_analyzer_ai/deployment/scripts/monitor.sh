#!/bin/bash
# Advanced Monitoring Script for Music Analyzer AI
# Monitors health, performance, and sends alerts
# Refactored with modular libraries

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"

# Initialize
init_common

# Configuration
readonly HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
readonly METRICS_URL="${METRICS_URL:-http://localhost:8000/metrics}"
readonly ALERT_EMAIL="${ALERT_EMAIL:-}"
readonly SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
readonly PROMETHEUS_URL="${PROMETHEUS_URL:-}"
readonly CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

# Thresholds
readonly CPU_THRESHOLD=80
readonly MEMORY_THRESHOLD=80
readonly DISK_THRESHOLD=80
readonly RESPONSE_TIME_THRESHOLD=2.0
readonly ERROR_RATE_THRESHOLD=5

# Send alert
send_alert() {
    local severity="${1:-warning}"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local full_message="[${timestamp}] [${severity^^}] Music Analyzer AI: ${message}"
    
    # Send to Slack
    if [ -n "${SLACK_WEBHOOK}" ]; then
        local emoji="⚠️"
        [ "${severity}" == "critical" ] && emoji="🚨"
        [ "${severity}" == "info" ] && emoji="ℹ️"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"${emoji} ${full_message}\"}" \
            "${SLACK_WEBHOOK}" &> /dev/null || true
    fi
    
    # Send email
    if [ -n "${ALERT_EMAIL}" ] && command -v mail &> /dev/null; then
        echo "${full_message}" | mail -s "Music Analyzer AI Alert [${severity^^}]" "${ALERT_EMAIL}" || true
    fi
    
    log_error "${message}"
}

# Check container health (using docker library)
check_container_health() {
    local container_name="${1:-music-analyzer-ai-backend}"
    
    local status=$(docker_get_container_status "${container_name}")
    
    if [ "${status}" != "running" ]; then
        send_alert "critical" "Container ${container_name} status: ${status}"
        return 1
    fi
    
    log_success "Container ${container_name} is healthy"
    return 0
}

# Check API health (using common library)
check_api_health() {
    if health_check "${HEALTH_URL}" 3 5; then
        return 0
    else
        send_alert "critical" "Health check failed"
        return 1
    fi
}

# Check response time
check_response_time() {
    local start_time=$(date +%s.%N)
    curl -f -s "${HEALTH_URL}" > /dev/null 2>&1 || return 1
    local end_time=$(date +%s.%N)
    local response_time=$(echo "${end_time} - ${start_time}" | bc)
    
    if (( $(echo "${response_time} > ${RESPONSE_TIME_THRESHOLD}" | bc -l) )); then
        send_alert "warning" "High response time: ${response_time}s (threshold: ${RESPONSE_TIME_THRESHOLD}s)"
        return 1
    fi
    
    log_info "Response time: ${response_time}s"
    return 0
}

# Check resource usage
check_resources() {
    local container_name="${1:-music-analyzer-ai-backend}"
    
    # CPU usage
    local cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" "${container_name}" 2>/dev/null | sed 's/%//' || echo "0")
    if (( $(echo "${cpu_usage} > ${CPU_THRESHOLD}" | bc -l) )); then
        send_alert "warning" "High CPU usage: ${cpu_usage}% (threshold: ${CPU_THRESHOLD}%)"
    fi
    
    # Memory usage
    local mem_usage=$(docker stats --no-stream --format "{{.MemPerc}}" "${container_name}" 2>/dev/null | sed 's/%//' || echo "0")
    if (( $(echo "${mem_usage} > ${MEMORY_THRESHOLD}" | bc -l) )); then
        send_alert "warning" "High memory usage: ${mem_usage}% (threshold: ${MEMORY_THRESHOLD}%)"
    fi
    
    log_info "CPU: ${cpu_usage}%, Memory: ${mem_usage}%"
}

# Check disk space
check_disk_space() {
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "${disk_usage}" -gt "${DISK_THRESHOLD}" ]; then
        send_alert "warning" "High disk usage: ${disk_usage}% (threshold: ${DISK_THRESHOLD}%)"
        return 1
    fi
    
    log_info "Disk usage: ${disk_usage}%"
    return 0
}

# Check error rate
check_error_rate() {
    if [ -z "${PROMETHEUS_URL}" ]; then
        return 0
    fi
    
    local error_rate=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    
    if (( $(echo "${error_rate} > ${ERROR_RATE_THRESHOLD}" | bc -l) )); then
        send_alert "critical" "High error rate: ${error_rate}% (threshold: ${ERROR_RATE_THRESHOLD}%)"
        return 1
    fi
    
    log_info "Error rate: ${error_rate}%"
    return 0
}

# Generate report
generate_report() {
    log_info "=== Music Analyzer AI Monitoring Report ==="
    log_info "Timestamp: $(date)"
    echo ""
    
    check_container_health "music-analyzer-ai-backend"
    check_container_health "music-analyzer-ai-frontend"
    check_api_health
    check_response_time
    check_resources "music-analyzer-ai-backend"
    check_disk_space
    check_error_rate
    
    log_info "=== End of Report ==="
}

# Continuous monitoring
monitor_continuous() {
    log_info "Starting continuous monitoring (interval: ${CHECK_INTERVAL}s)"
    
    while true; do
        generate_report
        sleep "${CHECK_INTERVAL}"
    done
}

# Main function
main() {
    case "${1:-report}" in
        health)
            check_container_health "music-analyzer-ai-backend" && check_api_health
            ;;
        resources)
            check_resources "music-analyzer-ai-backend"
            ;;
        disk)
            check_disk_space
            ;;
        continuous)
            monitor_continuous
            ;;
        report|*)
            generate_report
            ;;
    esac
}

main "$@"

