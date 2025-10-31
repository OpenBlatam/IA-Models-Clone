#!/bin/bash

# Production Monitoring Script for Blaze AI
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="blaze-ai"
DEPLOYMENT_DIR="deployment/kubernetes"
DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.prod.yml"
LOG_DIR="/var/log/blaze-ai"
ALERT_EMAIL="admin@blazeai.com"

# Thresholds
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85
DISK_THRESHOLD=90
RESPONSE_TIME_THRESHOLD=2000

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

check_prerequisites() {
    log_info "Checking monitoring prerequisites..."
    
    # Check if required tools are installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi
    
    # Create log directory
    mkdir -p $LOG_DIR
    
    log_info "Prerequisites check passed"
}

check_kubernetes_health() {
    log_info "Checking Kubernetes cluster health..."
    
    # Check cluster info
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    # Check nodes
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    READY_NODES=$(kubectl get nodes --no-headers | grep -c "Ready")
    
    if [ $READY_NODES -eq $NODE_COUNT ]; then
        log_info "All nodes are ready ($READY_NODES/$NODE_COUNT)"
    else
        log_warn "Some nodes are not ready ($READY_NODES/$NODE_COUNT)"
    fi
    
    # Check pods in blaze-ai namespace
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        POD_COUNT=$(kubectl get pods -n $NAMESPACE --no-headers | wc -l)
        RUNNING_PODS=$(kubectl get pods -n $NAMESPACE --no-headers | grep -c "Running")
        
        if [ $RUNNING_PODS -eq $POD_COUNT ]; then
            log_info "All pods are running ($RUNNING_PODS/$POD_COUNT)"
        else
            log_warn "Some pods are not running ($RUNNING_PODS/$POD_COUNT)"
            kubectl get pods -n $NAMESPACE
        fi
    else
        log_warn "Namespace $NAMESPACE not found"
    fi
}

check_docker_health() {
    log_info "Checking Docker services health..."
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        # Check if services are running
        RUNNING_SERVICES=$(docker-compose -f $DOCKER_COMPOSE_FILE ps --services --filter "status=running" | wc -l)
        TOTAL_SERVICES=$(docker-compose -f $DOCKER_COMPOSE_FILE config --services | wc -l)
        
        if [ $RUNNING_SERVICES -eq $TOTAL_SERVICES ]; then
            log_info "All Docker services are running ($RUNNING_SERVICES/$TOTAL_SERVICES)"
        else
            log_warn "Some Docker services are not running ($RUNNING_SERVICES/$TOTAL_SERVICES)"
            docker-compose -f $DOCKER_COMPOSE_FILE ps
        fi
        
        # Check service logs for errors
        docker-compose -f $DOCKER_COMPOSE_FILE logs --tail=50 | grep -i "error\|exception\|failed" | tail -10
    else
        log_warn "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
    fi
}

check_system_resources() {
    log_info "Checking system resources..."
    
    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    CPU_USAGE_INT=${CPU_USAGE%.*}
    
    if [ $CPU_USAGE_INT -gt $CPU_THRESHOLD ]; then
        log_warn "High CPU usage: ${CPU_USAGE}%"
    else
        log_info "CPU usage: ${CPU_USAGE}%"
    fi
    
    # Memory usage
    MEMORY_INFO=$(free | grep Mem)
    MEMORY_TOTAL=$(echo $MEMORY_INFO | awk '{print $2}')
    MEMORY_USED=$(echo $MEMORY_INFO | awk '{print $3}')
    MEMORY_USAGE=$((MEMORY_USED * 100 / MEMORY_TOTAL))
    
    if [ $MEMORY_USAGE -gt $MEMORY_THRESHOLD ]; then
        log_warn "High memory usage: ${MEMORY_USAGE}%"
    else
        log_info "Memory usage: ${MEMORY_USAGE}%"
    fi
    
    # Disk usage
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    
    if [ $DISK_USAGE -gt $DISK_THRESHOLD ]; then
        log_error "High disk usage: ${DISK_USAGE}%"
    else
        log_info "Disk usage: ${DISK_USAGE}%"
    fi
    
    # GPU usage (if available)
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Information:"
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    fi
}

check_application_health() {
    log_info "Checking application health endpoints..."
    
    # Check API health
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "API health check: OK"
    else
        log_error "API health check: FAILED"
    fi
    
    # Check Gradio health
    if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
        log_info "Gradio health check: OK"
    else
        log_warn "Gradio health check: FAILED"
    fi
    
    # Check metrics endpoint
    if curl -s -f http://localhost:8002/metrics > /dev/null 2>&1; then
        log_info "Metrics endpoint: OK"
    else
        log_warn "Metrics endpoint: FAILED"
    fi
    
    # Check response time
    RESPONSE_TIME=$(curl -s -w "%{time_total}" -o /dev/null http://localhost:8000/health)
    RESPONSE_TIME_MS=$(echo "$RESPONSE_TIME * 1000" | bc)
    
    if (( $(echo "$RESPONSE_TIME_MS > $RESPONSE_TIME_THRESHOLD" | bc -l) )); then
        log_warn "High response time: ${RESPONSE_TIME_MS}ms"
    else
        log_info "Response time: ${RESPONSE_TIME_MS}ms"
    fi
}

check_database_health() {
    log_info "Checking database health..."
    
    # Check PostgreSQL
    if command -v psql &> /dev/null; then
        if [ -n "$DB_PASSWORD" ]; then
            if PGPASSWORD=$DB_PASSWORD psql -h localhost -U blazeai -d blazeai -c "SELECT 1;" > /dev/null 2>&1; then
                log_info "PostgreSQL connection: OK"
            else
                log_error "PostgreSQL connection: FAILED"
            fi
        else
            log_warn "DB_PASSWORD not set, skipping PostgreSQL check"
        fi
    else
        log_warn "psql not installed, skipping PostgreSQL check"
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        if [ -n "$REDIS_PASSWORD" ]; then
            if redis-cli -a $REDIS_PASSWORD ping > /dev/null 2>&1; then
                log_info "Redis connection: OK"
            else
                log_error "Redis connection: FAILED"
            fi
        else
            log_warn "REDIS_PASSWORD not set, skipping Redis check"
        fi
    else
        log_warn "redis-cli not installed, skipping Redis check"
    fi
}

check_network_connectivity() {
    log_info "Checking network connectivity..."
    
    # Check external connectivity
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        log_info "External connectivity: OK"
    else
        log_error "External connectivity: FAILED"
    fi
    
    # Check DNS resolution
    if nslookup google.com > /dev/null 2>&1; then
        log_info "DNS resolution: OK"
    else
        log_error "DNS resolution: FAILED"
    fi
    
    # Check port availability
    PORTS=(8000 8001 8002 80 443)
    for port in "${PORTS[@]}"; do
        if netstat -tuln | grep ":$port " > /dev/null; then
            log_info "Port $port: LISTENING"
        else
            log_warn "Port $port: NOT LISTENING"
        fi
    done
}

check_logs() {
    log_info "Checking application logs..."
    
    # Check for recent errors in logs
    if [ -d "$LOG_DIR" ]; then
        ERROR_COUNT=$(find $LOG_DIR -name "*.log" -mtime -1 -exec grep -l "ERROR\|Exception\|Failed" {} \; | wc -l)
        
        if [ $ERROR_COUNT -gt 0 ]; then
            log_warn "Found $ERROR_COUNT log files with errors in the last 24 hours"
            
            # Show recent error lines
            find $LOG_DIR -name "*.log" -mtime -1 -exec grep -H "ERROR\|Exception\|Failed" {} \; | tail -10
        else
            log_info "No recent errors found in logs"
        fi
    else
        log_warn "Log directory not found: $LOG_DIR"
    fi
    
    # Check Docker logs for errors
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        DOCKER_ERRORS=$(docker-compose -f $DOCKER_COMPOSE_FILE logs --tail=100 | grep -c "ERROR\|Exception\|Failed")
        
        if [ $DOCKER_ERRORS -gt 0 ]; then
            log_warn "Found $DOCKER_ERRORS errors in Docker logs"
        else
            log_info "No recent errors in Docker logs"
        fi
    fi
}

check_security() {
    log_info "Checking security status..."
    
    # Check SSL certificate validity
    if [ -f "deployment/nginx/ssl/cert.pem" ]; then
        CERT_EXPIRY=$(openssl x509 -in deployment/nginx/ssl/cert.pem -enddate -noout | cut -d= -f2)
        CERT_DATE=$(date -d "$CERT_EXPIRY" +%s)
        CURRENT_DATE=$(date +%s)
        DAYS_LEFT=$(( (CERT_DATE - CURRENT_DATE) / 86400 ))
        
        if [ $DAYS_LEFT -lt 30 ]; then
            log_warn "SSL certificate expires in $DAYS_LEFT days"
        else
            log_info "SSL certificate expires in $DAYS_LEFT days"
        fi
    else
        log_warn "SSL certificate not found"
    fi
    
    # Check firewall status
    if command -v ufw &> /dev/null; then
        if ufw status | grep -q "Status: active"; then
            log_info "Firewall: ACTIVE"
        else
            log_warn "Firewall: INACTIVE"
        fi
    fi
    
    # Check for open ports
    OPEN_PORTS=$(netstat -tuln | grep LISTEN | wc -l)
    log_info "Open ports: $OPEN_PORTS"
}

generate_report() {
    log_info "Generating monitoring report..."
    
    REPORT_FILE="$LOG_DIR/monitoring_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > $REPORT_FILE << EOF
Blaze AI Monitoring Report
==========================
Generated: $(date)
Hostname: $(hostname)
Uptime: $(uptime)

System Resources:
- CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
- Memory Usage: $MEMORY_USAGE%
- Disk Usage: ${DISK_USAGE}%

Application Status:
- API Health: $(curl -s -f http://localhost:8000/health > /dev/null 2>&1 && echo "OK" || echo "FAILED")
- Gradio Health: $(curl -s -f http://localhost:8001/health > /dev/null 2>&1 && echo "OK" || echo "FAILED")
- Response Time: ${RESPONSE_TIME_MS}ms

Kubernetes Status:
- Namespace: $NAMESPACE
- Pods Running: $RUNNING_PODS/$POD_COUNT

Docker Status:
- Services Running: $RUNNING_SERVICES/$TOTAL_SERVICES

Security:
- SSL Certificate: $(if [ -f "deployment/nginx/ssl/cert.pem" ]; then echo "Found"; else echo "Not Found"; fi)
- Firewall: $(if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then echo "Active"; else echo "Inactive"; fi)

Recommendations:
$(if [ $CPU_USAGE_INT -gt $CPU_THRESHOLD ]; then echo "- Consider scaling up CPU resources"; fi)
$(if [ $MEMORY_USAGE -gt $MEMORY_THRESHOLD ]; then echo "- Consider scaling up memory resources"; fi)
$(if [ $DISK_USAGE -gt $DISK_THRESHOLD ]; then echo "- Clean up disk space or expand storage"; fi)
$(if [ $DAYS_LEFT -lt 30 ]; then echo "- Renew SSL certificate soon"; fi)
EOF
    
    log_info "Monitoring report generated: $REPORT_FILE"
}

send_alerts() {
    log_info "Checking if alerts need to be sent..."
    
    ALERT_NEEDED=false
    ALERT_MESSAGE=""
    
    # Check critical thresholds
    if [ $CPU_USAGE_INT -gt $CPU_THRESHOLD ]; then
        ALERT_NEEDED=true
        ALERT_MESSAGE="$ALERT_MESSAGE\n- High CPU usage: ${CPU_USAGE}%"
    fi
    
    if [ $MEMORY_USAGE -gt $MEMORY_THRESHOLD ]; then
        ALERT_NEEDED=true
        ALERT_MESSAGE="$ALERT_MESSAGE\n- High memory usage: ${MEMORY_USAGE}%"
    fi
    
    if [ $DISK_USAGE -gt $DISK_THRESHOLD ]; then
        ALERT_NEEDED=true
        ALERT_MESSAGE="$ALERT_MESSAGE\n- Critical disk usage: ${DISK_USAGE}%"
    fi
    
    if [ $DAYS_LEFT -lt 7 ]; then
        ALERT_NEEDED=true
        ALERT_MESSAGE="$ALERT_MESSAGE\n- SSL certificate expires in $DAYS_LEFT days"
    fi
    
    if [ "$ALERT_NEEDED" = true ]; then
        log_warn "Sending alert email..."
        echo -e "Subject: Blaze AI Alert - Critical Issues Detected\n\nCritical issues detected on $(hostname):$ALERT_MESSAGE" | \
        mail -s "Blaze AI Alert - Critical Issues Detected" $ALERT_EMAIL
    else
        log_info "No alerts needed"
    fi
}

# Main monitoring logic
main() {
    log_info "Starting Blaze AI production monitoring..."
    
    check_prerequisites
    
    # Perform health checks
    check_kubernetes_health
    check_docker_health
    check_system_resources
    check_application_health
    check_database_health
    check_network_connectivity
    check_logs
    check_security
    
    # Generate report and send alerts
    generate_report
    send_alerts
    
    log_info "Monitoring completed successfully!"
}

# Run main function
main "$@"
