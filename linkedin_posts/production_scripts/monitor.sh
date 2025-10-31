#!/bin/bash

# LinkedIn Posts Production Monitoring Script
# ==========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_NAME="linkedin-posts"
DOCKER_COMPOSE_FILE="docker-compose.prod.yml"
LOG_DIR="logs"
ALERT_EMAIL="admin@yourdomain.com"
ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Thresholds
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85
DISK_THRESHOLD=90
API_RESPONSE_THRESHOLD=2000  # milliseconds

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

send_alert() {
    local level="$1"
    local message="$2"
    
    # Log alert
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [$level] $message" >> "$LOG_DIR/alerts.log"
    
    # Send email alert
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "[$level] LinkedIn Posts Alert" "$ALERT_EMAIL"
    fi
    
    # Send webhook alert
    if [ -n "$ALERT_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"[$level] LinkedIn Posts Alert: $message\"}" \
            "$ALERT_WEBHOOK" &> /dev/null || true
    fi
}

check_docker_services() {
    log_info "Checking Docker services..."
    
    # Get service status
    local services_status=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps --format json)
    local failed_services=()
    
    # Check each service
    while IFS= read -r line; do
        local service_name=$(echo "$line" | jq -r '.Service')
        local state=$(echo "$line" | jq -r '.State')
        
        if [ "$state" != "running" ]; then
            failed_services+=("$service_name ($state)")
        fi
    done <<< "$services_status"
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        local message="Failed services: ${failed_services[*]}"
        log_error "$message"
        send_alert "ERROR" "$message"
        return 1
    else
        log_success "All Docker services are running"
        return 0
    fi
}

check_api_health() {
    log_info "Checking API health..."
    
    local start_time=$(date +%s%3N)
    local response=$(curl -s -w "%{http_code}" -o /tmp/api_response http://localhost:8000/health)
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    if [ "$response" = "200" ]; then
        log_success "API health check passed (${response_time}ms)"
        
        # Check response time threshold
        if [ $response_time -gt $API_RESPONSE_THRESHOLD ]; then
            local message="API response time is slow: ${response_time}ms"
            log_warning "$message"
            send_alert "WARNING" "$message"
        fi
        
        return 0
    else
        local message="API health check failed with status $response"
        log_error "$message"
        send_alert "ERROR" "$message"
        return 1
    fi
}

check_database_health() {
    log_info "Checking database health..."
    
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U linkedin_user &> /dev/null; then
        log_success "Database health check passed"
        return 0
    else
        local message="Database health check failed"
        log_error "$message"
        send_alert "ERROR" "$message"
        return 1
    fi
}

check_redis_health() {
    log_info "Checking Redis health..."
    
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis health check passed"
        return 0
    else
        local message="Redis health check failed"
        log_error "$message"
        send_alert "ERROR" "$message"
        return 1
    fi
}

check_system_resources() {
    log_info "Checking system resources..."
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )); then
        local message="High CPU usage: ${cpu_usage}%"
        log_warning "$message"
        send_alert "WARNING" "$message"
    fi
    
    # Memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
        local message="High memory usage: ${memory_usage}%"
        log_warning "$message"
        send_alert "WARNING" "$message"
    fi
    
    # Disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    if [ "$disk_usage" -gt "$DISK_THRESHOLD" ]; then
        local message="High disk usage: ${disk_usage}%"
        log_warning "$message"
        send_alert "WARNING" "$message"
    fi
    
    log_success "System resource check completed"
}

check_container_resources() {
    log_info "Checking container resources..."
    
    # Get container stats
    local containers=("linkedin-posts-api" "linkedin-posts-postgres" "linkedin-posts-redis")
    
    for container in "${containers[@]}"; do
        if docker ps | grep -q "$container"; then
            local stats=$(docker stats "$container" --no-stream --format "table {{.CPUPerc}}\t{{.MemPerc}}\t{{.MemUsage}}")
            local cpu_perc=$(echo "$stats" | tail -1 | awk '{print $1}' | cut -d'%' -f1)
            local mem_perc=$(echo "$stats" | tail -1 | awk '{print $2}' | cut -d'%' -f1)
            
            if (( $(echo "$cpu_perc > $CPU_THRESHOLD" | bc -l) )); then
                local message="High CPU usage in $container: ${cpu_perc}%"
                log_warning "$message"
                send_alert "WARNING" "$message"
            fi
            
            if (( $(echo "$mem_perc > $MEMORY_THRESHOLD" | bc -l) )); then
                local message="High memory usage in $container: ${mem_perc}%"
                log_warning "$message"
                send_alert "WARNING" "$message"
            fi
        fi
    done
    
    log_success "Container resource check completed"
}

check_logs_for_errors() {
    log_info "Checking logs for errors..."
    
    # Check for recent errors in API logs
    local error_count=$(docker-compose -f "$DOCKER_COMPOSE_FILE" logs --since=1h linkedin-posts-api 2>/dev/null | grep -i "error\|exception\|traceback" | wc -l)
    
    if [ "$error_count" -gt 10 ]; then
        local message="High error count in API logs: $error_count errors in last hour"
        log_warning "$message"
        send_alert "WARNING" "$message"
    fi
    
    # Check for recent errors in database logs
    local db_error_count=$(docker-compose -f "$DOCKER_COMPOSE_FILE" logs --since=1h postgres 2>/dev/null | grep -i "error\|fatal" | wc -l)
    
    if [ "$db_error_count" -gt 5 ]; then
        local message="High error count in database logs: $db_error_count errors in last hour"
        log_warning "$message"
        send_alert "WARNING" "$message"
    fi
    
    log_success "Log error check completed"
}

check_backup_status() {
    log_info "Checking backup status..."
    
    # Check if backup directory exists
    if [ ! -d "backups" ]; then
        local message="Backup directory not found"
        log_error "$message"
        send_alert "ERROR" "$message"
        return 1
    fi
    
    # Check last backup time
    local last_backup=$(ls -t backups/*.sql.gz 2>/dev/null | head -n 1)
    if [ -z "$last_backup" ]; then
        local message="No backups found"
        log_error "$message"
        send_alert "ERROR" "$message"
        return 1
    fi
    
    local backup_age=$(( $(date +%s) - $(stat -c %Y "$last_backup") ))
    local backup_age_hours=$((backup_age / 3600))
    
    if [ "$backup_age_hours" -gt 24 ]; then
        local message="Backup is old: ${backup_age_hours} hours"
        log_warning "$message"
        send_alert "WARNING" "$message"
    else
        log_success "Backup status OK (${backup_age_hours} hours old)"
    fi
}

generate_monitoring_report() {
    log_info "Generating monitoring report..."
    
    local report_file="$LOG_DIR/monitoring_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "LinkedIn Posts Monitoring Report"
        echo "Generated: $(date)"
        echo "=================================="
        echo
        echo "System Status:"
        echo "-------------"
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        echo
        echo "System Resources:"
        echo "----------------"
        echo "CPU Usage: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}')"
        echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
        echo "Disk Usage: $(df / | tail -1 | awk '{print $5}')"
        echo
        echo "Container Resources:"
        echo "-------------------"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemPerc}}\t{{.MemUsage}}"
        echo
        echo "Recent Alerts:"
        echo "-------------"
        tail -10 "$LOG_DIR/alerts.log" 2>/dev/null || echo "No alerts found"
        echo
        echo "API Health:"
        echo "----------"
        curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "API health check failed"
    } > "$report_file"
    
    log_success "Monitoring report generated: $report_file"
}

# Main monitoring function
monitor() {
    log_info "Starting LinkedIn Posts monitoring..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Run all checks
    local overall_status=0
    
    check_docker_services || overall_status=1
    check_api_health || overall_status=1
    check_database_health || overall_status=1
    check_redis_health || overall_status=1
    check_system_resources
    check_container_resources
    check_logs_for_errors
    check_backup_status || overall_status=1
    
    # Generate report
    generate_monitoring_report
    
    if [ $overall_status -eq 0 ]; then
        log_success "All monitoring checks passed"
    else
        log_error "Some monitoring checks failed"
    fi
    
    return $overall_status
}

# Continuous monitoring
monitor_continuous() {
    log_info "Starting continuous monitoring..."
    
    while true; do
        monitor
        sleep 300  # Check every 5 minutes
    done
}

# Command line arguments
case "${1:-monitor}" in
    "monitor")
        monitor
        ;;
    "continuous")
        monitor_continuous
        ;;
    "report")
        generate_monitoring_report
        ;;
    "health")
        check_api_health
        ;;
    "resources")
        check_system_resources
        check_container_resources
        ;;
    *)
        echo "Usage: $0 {monitor|continuous|report|health|resources}"
        echo
        echo "Commands:"
        echo "  monitor     - Run one-time monitoring check (default)"
        echo "  continuous  - Run continuous monitoring"
        echo "  report      - Generate monitoring report"
        echo "  health      - Check API health only"
        echo "  resources   - Check system and container resources"
        exit 1
        ;;
esac 