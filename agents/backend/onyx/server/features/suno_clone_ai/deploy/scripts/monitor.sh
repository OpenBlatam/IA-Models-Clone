#!/bin/bash
# Monitoring script for Suno Clone AI
# Checks health, metrics, and sends alerts

set -euo pipefail

# Configuration
readonly CONTAINER_NAME="suno-clone-ai"
readonly HEALTH_URL="http://localhost:8020/health"
readonly METRICS_URL="http://localhost:8020/metrics"
readonly ALERT_EMAIL="${ALERT_EMAIL:-}"
readonly SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check container status
check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container ${CONTAINER_NAME} is not running"
        send_alert "Container ${CONTAINER_NAME} is not running"
        return 1
    fi
    log_info "Container is running"
    return 0
}

# Check health endpoint
check_health() {
    local response=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}" || echo "000")
    
    if [ "${response}" != "200" ]; then
        log_error "Health check failed (HTTP ${response})"
        send_alert "Health check failed with HTTP ${response}"
        return 1
    fi
    
    log_info "Health check passed"
    return 0
}

# Check metrics
check_metrics() {
    if curl -s "${METRICS_URL}" &> /dev/null; then
        log_info "Metrics endpoint accessible"
        
        # Extract key metrics
        local cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" "${CONTAINER_NAME}" | sed 's/%//')
        local mem_usage=$(docker stats --no-stream --format "{{.MemUsage}}" "${CONTAINER_NAME}")
        
        log_info "CPU Usage: ${cpu_usage}%"
        log_info "Memory Usage: ${mem_usage}"
        
        # Alert if CPU > 80%
        if (( $(echo "${cpu_usage} > 80" | bc -l) )); then
            log_warn "High CPU usage: ${cpu_usage}%"
            send_alert "High CPU usage: ${cpu_usage}%"
        fi
    else
        log_warn "Metrics endpoint not accessible"
    fi
}

# Check disk space
check_disk_space() {
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    log_info "Disk usage: ${disk_usage}%"
    
    if [ "${disk_usage}" -gt 80 ]; then
        log_warn "High disk usage: ${disk_usage}%"
        send_alert "High disk usage: ${disk_usage}%"
    fi
}

# Check logs for errors
check_logs() {
    local error_count=$(docker logs "${CONTAINER_NAME}" --since 5m 2>&1 | grep -i "error\|exception\|fatal" | wc -l)
    
    if [ "${error_count}" -gt 10 ]; then
        log_warn "High error count in logs: ${error_count} errors in last 5 minutes"
        send_alert "High error count: ${error_count} errors in last 5 minutes"
    fi
}

# Send alert
send_alert() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local full_message="[${timestamp}] Suno Clone AI Alert: ${message}"
    
    # Send to Slack if configured
    if [ -n "${SLACK_WEBHOOK}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"${full_message}\"}" \
            "${SLACK_WEBHOOK}" &> /dev/null || true
    fi
    
    # Send email if configured
    if [ -n "${ALERT_EMAIL}" ] && command -v mail &> /dev/null; then
        echo "${full_message}" | mail -s "Suno Clone AI Alert" "${ALERT_EMAIL}" || true
    fi
    
    log_error "${message}"
}

# Generate report
generate_report() {
    log_info "=== Suno Clone AI Monitoring Report ==="
    log_info "Timestamp: $(date)"
    echo ""
    
    check_container
    check_health
    check_metrics
    check_disk_space
    check_logs
    
    log_info "=== End of Report ==="
}

# Main function
main() {
    case "${1:-report}" in
        health)
            check_container && check_health
            ;;
        metrics)
            check_metrics
            ;;
        disk)
            check_disk_space
            ;;
        logs)
            check_logs
            ;;
        report|*)
            generate_report
            ;;
    esac
}

main "$@"




