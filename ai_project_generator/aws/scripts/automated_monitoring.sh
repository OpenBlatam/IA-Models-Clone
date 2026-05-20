#!/bin/bash

###############################################################################
# Automated Monitoring Script for AI Project Generator
# Monitors application health, resources, and sends alerts
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PORT="${APP_PORT:-8020}"
HEALTH_URL="http://localhost:${APP_PORT}/health"
METRICS_URL="http://localhost:${APP_PORT}/metrics"
LOG_FILE="${LOG_FILE:-/var/log/monitoring.log}"
ALERT_EMAIL="${ALERT_EMAIL:-}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"

# Thresholds
CPU_THRESHOLD="${CPU_THRESHOLD:-80}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-85}"
DISK_THRESHOLD="${DISK_THRESHOLD:-90}"
RESPONSE_TIME_THRESHOLD="${RESPONSE_TIME_THRESHOLD:-5000}" # milliseconds

###############################################################################
# Helper Functions
###############################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

send_alert() {
    local subject="$1"
    local message="$2"
    
    # Send via SNS if configured
    if [ -n "${SNS_TOPIC_ARN}" ] && command -v aws &> /dev/null; then
        aws sns publish \
            --topic-arn "${SNS_TOPIC_ARN}" \
            --subject "${subject}" \
            --message "${message}" \
            > /dev/null 2>&1 || log_warn "Failed to send SNS alert"
    fi
    
    # Send via email if configured
    if [ -n "${ALERT_EMAIL}" ] && command -v mail &> /dev/null; then
        echo "${message}" | mail -s "${subject}" "${ALERT_EMAIL}" 2>/dev/null || true
    fi
    
    log_warn "ALERT: ${subject} - ${message}"
}

###############################################################################
# Monitoring Functions
###############################################################################

check_application_health() {
    log_info "Checking application health..."
    
    local response_code
    local response_time
    
    response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${HEALTH_URL}" 2>/dev/null || echo "000")
    response_time=$(curl -s -o /dev/null -w "%{time_total}" --max-time 10 "${HEALTH_URL}" 2>/dev/null || echo "0")
    response_time_ms=$(echo "${response_time} * 1000" | bc | cut -d. -f1)
    
    if [ "${response_code}" != "200" ]; then
        send_alert "Application Health Check Failed" \
            "Health endpoint returned status code: ${response_code}"
        return 1
    fi
    
    if [ "${response_time_ms}" -gt "${RESPONSE_TIME_THRESHOLD}" ]; then
        send_alert "Application Slow Response" \
            "Health check response time: ${response_time_ms}ms (threshold: ${RESPONSE_TIME_THRESHOLD}ms)"
    fi
    
    log_info "Application health: OK (${response_code}, ${response_time_ms}ms)"
    return 0
}

check_cpu_usage() {
    log_info "Checking CPU usage..."
    
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    cpu_usage_int=$(echo "${cpu_usage}" | cut -d. -f1)
    
    if [ "${cpu_usage_int}" -gt "${CPU_THRESHOLD}" ]; then
        send_alert "High CPU Usage" \
            "CPU usage: ${cpu_usage}% (threshold: ${CPU_THRESHOLD}%)"
        return 1
    fi
    
    log_info "CPU usage: ${cpu_usage}%"
    return 0
}

check_memory_usage() {
    log_info "Checking memory usage..."
    
    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    
    if [ "${memory_usage}" -gt "${MEMORY_THRESHOLD}" ]; then
        send_alert "High Memory Usage" \
            "Memory usage: ${memory_usage}% (threshold: ${MEMORY_THRESHOLD}%)"
        return 1
    fi
    
    log_info "Memory usage: ${memory_usage}%"
    return 0
}

check_disk_usage() {
    log_info "Checking disk usage..."
    
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "${disk_usage}" -gt "${DISK_THRESHOLD}" ]; then
        send_alert "High Disk Usage" \
            "Disk usage: ${disk_usage}% (threshold: ${DISK_THRESHOLD}%)"
        return 1
    fi
    
    log_info "Disk usage: ${disk_usage}%"
    return 0
}

check_docker_containers() {
    log_info "Checking Docker containers..."
    
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not found, skipping container check"
        return 0
    fi
    
    local stopped_containers
    stopped_containers=$(docker ps -a --filter "status=exited" --format "{{.Names}}" | wc -l)
    
    if [ "${stopped_containers}" -gt 0 ]; then
        local container_list
        container_list=$(docker ps -a --filter "status=exited" --format "{{.Names}}")
        send_alert "Docker Containers Stopped" \
            "Found ${stopped_containers} stopped container(s): ${container_list}"
        return 1
    fi
    
    log_info "Docker containers: All running"
    return 0
}

check_redis() {
    log_info "Checking Redis..."
    
    if ! command -v redis-cli &> /dev/null; then
        log_warn "Redis CLI not found, skipping Redis check"
        return 0
    fi
    
    if ! redis-cli ping > /dev/null 2>&1; then
        send_alert "Redis Not Responding" \
            "Redis server is not responding to PING command"
        return 1
    fi
    
    log_info "Redis: OK"
    return 0
}

check_nginx() {
    log_info "Checking Nginx..."
    
    if ! systemctl is-active --quiet nginx; then
        send_alert "Nginx Not Running" \
            "Nginx service is not running"
        return 1
    fi
    
    if ! nginx -t > /dev/null 2>&1; then
        send_alert "Nginx Configuration Error" \
            "Nginx configuration test failed"
        return 1
    fi
    
    log_info "Nginx: OK"
    return 0
}

collect_metrics() {
    log_info "Collecting application metrics..."
    
    if curl -s --max-time 10 "${METRICS_URL}" > /tmp/metrics.txt 2>/dev/null; then
        log_info "Metrics collected successfully"
        # You can process metrics here or send to CloudWatch
        if command -v aws &> /dev/null; then
            # Example: Send custom metrics to CloudWatch
            # aws cloudwatch put-metric-data --namespace "AIProjectGenerator" ...
        fi
    else
        log_warn "Failed to collect metrics"
    fi
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "Starting automated monitoring checks..."
    
    local exit_code=0
    
    check_application_health || exit_code=1
    check_cpu_usage || exit_code=1
    check_memory_usage || exit_code=1
    check_disk_usage || exit_code=1
    check_docker_containers || exit_code=1
    check_redis || exit_code=1
    check_nginx || exit_code=1
    collect_metrics
    
    if [ $exit_code -eq 0 ]; then
        log_info "✅ All monitoring checks passed"
    else
        log_error "❌ Some monitoring checks failed"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"

