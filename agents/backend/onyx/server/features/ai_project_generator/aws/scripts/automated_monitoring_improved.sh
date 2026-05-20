#!/bin/bash

###############################################################################
# Improved Automated Monitoring Script for AI Project Generator
# Enhanced monitoring with better metrics, alerting, and CloudWatch integration
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh" 2>/dev/null || {
    echo "Error: common_functions.sh not found" >&2
    exit 1
}

# Configuration
APP_PORT="${APP_PORT:-8020}"
HEALTH_URL="http://localhost:${APP_PORT}/health"
METRICS_URL="http://localhost:${APP_PORT}/metrics"
STATUS_URL="http://localhost:${APP_PORT}/api/v1/status"
LOG_FILE="${LOG_FILE:-/var/log/monitoring.log}"
ALERT_EMAIL="${ALERT_EMAIL:-}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"
CLOUDWATCH_NAMESPACE="${CLOUDWATCH_NAMESPACE:-AIProjectGenerator/Monitoring}"
ENABLE_CLOUDWATCH="${ENABLE_CLOUDWATCH:-true}"

# Thresholds
CPU_THRESHOLD="${CPU_THRESHOLD:-80}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-85}"
DISK_THRESHOLD="${DISK_THRESHOLD:-90}"
RESPONSE_TIME_THRESHOLD="${RESPONSE_TIME_THRESHOLD:-5000}" # milliseconds
ERROR_RATE_THRESHOLD="${ERROR_RATE_THRESHOLD:-5}" # percentage

# Monitoring state
MONITORING_START_TIME=$(date +%s)
ALERT_COUNT=0
CHECK_COUNT=0
FAILED_CHECKS=0

###############################################################################
# Monitoring Functions
###############################################################################

check_application_health() {
    log_info "Checking application health..."
    CHECK_COUNT=$((CHECK_COUNT + 1))
    
    local response_code
    local response_time_ms
    local health_status="unknown"
    
    # Get HTTP status
    response_code=$(get_http_status "${HEALTH_URL}" 10)
    
    # Get response time
    local response_time
    response_time=$(get_response_time "${HEALTH_URL}" 10)
    response_time_ms=$(echo "${response_time} * 1000" | bc | cut -d. -f1)
    
    # Check status endpoint for detailed health
    if check_url "${STATUS_URL}" 10 1; then
        health_status=$(curl -s --max-time 10 "${STATUS_URL}" | \
            jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
    fi
    
    # Send metrics to CloudWatch
    if [ "${ENABLE_CLOUDWATCH}" = "true" ]; then
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "HealthCheckResponseTime" \
            "${response_time_ms}" "Milliseconds"
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "HealthCheckStatusCode" \
            "${response_code}" "None"
    fi
    
    if [ "${response_code}" != "200" ]; then
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        ALERT_COUNT=$((ALERT_COUNT + 1))
        send_alert "Application Health Check Failed" \
            "Health endpoint returned status code: ${response_code}\nResponse time: ${response_time_ms}ms\nHealth status: ${health_status}"
        return 1
    fi
    
    if [ "${response_time_ms}" -gt "${RESPONSE_TIME_THRESHOLD}" ]; then
        ALERT_COUNT=$((ALERT_COUNT + 1))
        send_alert "Application Slow Response" \
            "Health check response time: ${response_time_ms}ms (threshold: ${RESPONSE_TIME_THRESHOLD}ms)\nStatus code: ${response_code}"
    fi
    
    log_success "Application health: OK (${response_code}, ${response_time_ms}ms, ${health_status})"
    return 0
}

check_system_resources() {
    log_info "Checking system resources..."
    
    # CPU Usage
    local cpu_usage
    cpu_usage=$(get_cpu_usage)
    cpu_usage_int=$(echo "${cpu_usage}" | cut -d. -f1)
    
    if [ "${ENABLE_CLOUDWATCH}" = "true" ]; then
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "CPUUtilization" \
            "${cpu_usage}" "Percent"
    fi
    
    if [ "${cpu_usage_int}" -gt "${CPU_THRESHOLD}" ]; then
        ALERT_COUNT=$((ALERT_COUNT + 1))
        send_alert "High CPU Usage" \
            "CPU usage: ${cpu_usage}% (threshold: ${CPU_THRESHOLD}%)"
        return 1
    fi
    
    log_info "CPU usage: ${cpu_usage}%"
    
    # Memory Usage
    local memory_usage
    memory_usage=$(get_memory_usage)
    
    if [ "${ENABLE_CLOUDWATCH}" = "true" ]; then
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "MemoryUtilization" \
            "${memory_usage}" "Percent"
    fi
    
    if [ "${memory_usage}" -gt "${MEMORY_THRESHOLD}" ]; then
        ALERT_COUNT=$((ALERT_COUNT + 1))
        send_alert "High Memory Usage" \
            "Memory usage: ${memory_usage}% (threshold: ${MEMORY_THRESHOLD}%)"
        return 1
    fi
    
    log_info "Memory usage: ${memory_usage}%"
    
    # Disk Usage
    local disk_usage
    disk_usage=$(get_disk_usage "/")
    local disk_available
    disk_available=$(get_disk_available "/")
    
    if [ "${ENABLE_CLOUDWATCH}" = "true" ]; then
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "DiskUtilization" \
            "${disk_usage}" "Percent"
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "DiskAvailable" \
            "${disk_available}" "Gigabytes"
    fi
    
    if [ "${disk_usage}" -gt "${DISK_THRESHOLD}" ]; then
        ALERT_COUNT=$((ALERT_COUNT + 1))
        send_alert "High Disk Usage" \
            "Disk usage: ${disk_usage}% (${disk_available}GB available, threshold: ${DISK_THRESHOLD}%)"
        return 1
    fi
    
    log_info "Disk usage: ${disk_usage}% (${disk_available}GB available)"
    return 0
}

check_docker_containers() {
    log_info "Checking Docker containers..."
    
    if ! check_command "docker"; then
        log_warn "Docker not found, skipping container check"
        return 0
    fi
    
    local stopped_containers
    stopped_containers=$(docker ps -a --filter "status=exited" --format "{{.Names}}" | wc -l)
    local total_containers
    total_containers=$(docker ps -a --format "{{.Names}}" | wc -l)
    local running_containers
    running_containers=$(docker ps --format "{{.Names}}" | wc -l)
    
    if [ "${ENABLE_CLOUDWATCH}" = "true" ]; then
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "DockerContainersTotal" \
            "${total_containers}" "Count"
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "DockerContainersRunning" \
            "${running_containers}" "Count"
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "DockerContainersStopped" \
            "${stopped_containers}" "Count"
    fi
    
    if [ "${stopped_containers}" -gt 0 ]; then
        local container_list
        container_list=$(docker ps -a --filter "status=exited" --format "{{.Names}}" | tr '\n' ', ')
        ALERT_COUNT=$((ALERT_COUNT + 1))
        send_alert "Docker Containers Stopped" \
            "Found ${stopped_containers} stopped container(s) out of ${total_containers} total:\n${container_list}"
        return 1
    fi
    
    # Check container health
    local unhealthy_containers=0
    while IFS= read -r container; do
        if [ -n "${container}" ]; then
            local health_status
            health_status=$(get_docker_container_health "${container}")
            if [ "${health_status}" = "unhealthy" ]; then
                unhealthy_containers=$((unhealthy_containers + 1))
                ALERT_COUNT=$((ALERT_COUNT + 1))
                send_alert "Docker Container Unhealthy" \
                    "Container ${container} is unhealthy"
            fi
        fi
    done < <(docker ps --format "{{.Names}}")
    
    if [ $unhealthy_containers -eq 0 ]; then
        log_success "Docker containers: All running and healthy (${running_containers}/${total_containers})"
    fi
    
    return 0
}

check_services() {
    log_info "Checking system services..."
    
    # Redis
    if check_command "redis-cli"; then
        if redis-cli ping > /dev/null 2>&1; then
            log_success "Redis: OK"
        else
            ALERT_COUNT=$((ALERT_COUNT + 1))
            send_alert "Redis Not Responding" \
                "Redis server is not responding to PING command"
            return 1
        fi
    fi
    
    # Nginx
    if systemctl is-active --quiet nginx; then
        if nginx -t > /dev/null 2>&1; then
            log_success "Nginx: OK"
        else
            ALERT_COUNT=$((ALERT_COUNT + 1))
            send_alert "Nginx Configuration Error" \
                "Nginx configuration test failed"
            return 1
        fi
    else
        ALERT_COUNT=$((ALERT_COUNT + 1))
        send_alert "Nginx Not Running" \
            "Nginx service is not running"
        return 1
    fi
    
    return 0
}

collect_application_metrics() {
    log_info "Collecting application metrics..."
    
    if check_url "${METRICS_URL}" 10 1; then
        local metrics_file="/tmp/metrics_$(date +%s).txt"
        if curl -s --max-time 10 "${METRICS_URL}" > "${metrics_file}" 2>/dev/null; then
            log_debug "Metrics collected successfully"
            
            # Parse and send key metrics to CloudWatch
            if [ "${ENABLE_CLOUDWATCH}" = "true" ] && command -v grep &> /dev/null; then
                # Example: Extract specific metrics (adjust based on your metrics format)
                # This is a placeholder - adjust based on your actual metrics format
                local request_count
                request_count=$(grep -oP 'http_requests_total \K[0-9]+' "${metrics_file}" 2>/dev/null || echo "0")
                if [ -n "${request_count}" ] && [ "${request_count}" != "0" ]; then
                    send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "RequestCount" \
                        "${request_count}" "Count"
                fi
            fi
            
            rm -f "${metrics_file}"
        else
            log_warn "Failed to collect metrics"
        fi
    else
        log_warn "Metrics endpoint not accessible"
    fi
}

send_alert() {
    local subject="$1"
    local message="$2"
    
    # Send via SNS if configured
    if [ -n "${SNS_TOPIC_ARN}" ]; then
        send_sns_alert "${SNS_TOPIC_ARN}" "${subject}" "${message}" || true
    fi
    
    # Send via email if configured
    if [ -n "${ALERT_EMAIL}" ] && check_command "mail"; then
        echo -e "${message}" | mail -s "${subject}" "${ALERT_EMAIL}" 2>/dev/null || true
    fi
    
    log_warn "ALERT: ${subject}"
    log_warn "Details: ${message}"
}

generate_monitoring_report() {
    local end_time=$(date +%s)
    local duration=$((end_time - MONITORING_START_TIME))
    local success_rate
    success_rate=$(echo "scale=2; (${CHECK_COUNT} - ${FAILED_CHECKS}) * 100 / ${CHECK_COUNT}" | bc 2>/dev/null || echo "0")
    
    local report_file="/tmp/monitoring_report_$(date +%Y%m%d_%H%M%S).txt"
    generate_report "${report_file}" "Monitoring Report" \
        "Duration: $(format_duration ${duration})\n" \
        "Checks Performed: ${CHECK_COUNT}\n" \
        "Failed Checks: ${FAILED_CHECKS}\n" \
        "Success Rate: ${success_rate}%\n" \
        "Alerts Generated: ${ALERT_COUNT}\n" \
        "CPU Usage: $(get_cpu_usage)%\n" \
        "Memory Usage: $(get_memory_usage)%\n" \
        "Disk Usage: $(get_disk_usage /)%\n"
    
    log_info "Monitoring report: ${report_file}"
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "=========================================="
    log_info "Starting automated monitoring checks"
    log_info "=========================================="
    
    local exit_code=0
    
    check_application_health || exit_code=1
    check_system_resources || exit_code=1
    check_docker_containers || exit_code=1
    check_services || exit_code=1
    collect_application_metrics
    
    # Generate report
    generate_monitoring_report
    
    # Send summary metrics
    if [ "${ENABLE_CLOUDWATCH}" = "true" ]; then
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "MonitoringChecksTotal" \
            "${CHECK_COUNT}" "Count"
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "MonitoringChecksFailed" \
            "${FAILED_CHECKS}" "Count"
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "AlertsGenerated" \
            "${ALERT_COUNT}" "Count"
    fi
    
    log_info "=========================================="
    if [ $exit_code -eq 0 ]; then
        log_success "All monitoring checks passed"
    else
        log_error "Some monitoring checks failed"
    fi
    log_info "=========================================="
    
    exit $exit_code
}

# Run main function
main "$@"

