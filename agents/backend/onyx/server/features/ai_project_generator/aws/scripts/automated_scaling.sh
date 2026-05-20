#!/bin/bash

###############################################################################
# Automated Scaling Script for AI Project Generator
# Automatically scales EC2 instances based on metrics
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh" 2>/dev/null || {
    echo "Error: common_functions.sh not found" >&2
    exit 1
}

# Configuration
LOG_FILE="${LOG_FILE:-/var/log/auto-scaling.log}"
CLOUDWATCH_NAMESPACE="${CLOUDWATCH_NAMESPACE:-AIProjectGenerator/Scaling}"
AUTO_SCALING_GROUP="${AUTO_SCALING_GROUP:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Scaling thresholds
CPU_SCALE_UP_THRESHOLD="${CPU_SCALE_UP_THRESHOLD:-75}"
CPU_SCALE_DOWN_THRESHOLD="${CPU_SCALE_DOWN_THRESHOLD:-25}"
MEMORY_SCALE_UP_THRESHOLD="${MEMORY_SCALE_UP_THRESHOLD:-80}"
MEMORY_SCALE_DOWN_THRESHOLD="${MEMORY_SCALE_DOWN_THRESHOLD:-30}"
REQUEST_RATE_SCALE_UP="${REQUEST_RATE_SCALE_UP:-1000}" # requests per minute
REQUEST_RATE_SCALE_DOWN="${REQUEST_RATE_SCALE_DOWN:-100}" # requests per minute

# Scaling limits
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
SCALE_UP_COOLDOWN="${SCALE_UP_COOLDOWN:-300}" # seconds
SCALE_DOWN_COOLDOWN="${SCALE_DOWN_COOLDOWN:-600}" # seconds

# State tracking
LAST_SCALE_ACTION=""
LAST_SCALE_TIME=0
CURRENT_CAPACITY=0

###############################################################################
# Helper Functions
###############################################################################

get_current_capacity() {
    if [ -z "${AUTO_SCALING_GROUP}" ]; then
        log_warn "Auto Scaling Group not specified"
        return 1
    fi
    
    CURRENT_CAPACITY=$(aws autoscaling describe-auto-scaling-groups \
        --auto-scaling-group-names "${AUTO_SCALING_GROUP}" \
        --region "${AWS_REGION}" \
        --query 'AutoScalingGroups[0].DesiredCapacity' \
        --output text 2>/dev/null || echo "0")
    
    log_debug "Current capacity: ${CURRENT_CAPACITY}"
    echo "${CURRENT_CAPACITY}"
}

get_average_cpu() {
    local period="${1:-300}" # 5 minutes
    local end_time=$(date -u +%Y-%m-%dT%H:%M:%S)
    local start_time=$(date -u -d "${period} seconds ago" +%Y-%m-%dT%H:%M:%S 2>/dev/null || \
        date -u -v-${period}S +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "")
    
    if [ -z "${start_time}" ]; then
        # Fallback to current CPU
        get_cpu_usage
        return 0
    fi
    
    # Get CPU from CloudWatch
    local cpu_avg=$(aws cloudwatch get-metric-statistics \
        --namespace AWS/EC2 \
        --metric-name CPUUtilization \
        --dimensions Name=AutoScalingGroupName,Value="${AUTO_SCALING_GROUP}" \
        --start-time "${start_time}" \
        --end-time "${end_time}" \
        --period 60 \
        --statistics Average \
        --region "${AWS_REGION}" \
        --query 'Datapoints[*].Average' \
        --output text 2>/dev/null | \
        awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "0"}')
    
    echo "${cpu_avg:-0}"
}

get_average_memory() {
    # Get memory from CloudWatch custom metrics
    local period="${1:-300}"
    local end_time=$(date -u +%Y-%m-%dT%H:%M:%S)
    local start_time=$(date -u -d "${period} seconds ago" +%Y-%m-%dT%H:%M:%S 2>/dev/null || \
        date -u -v-${period}S +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "")
    
    if [ -z "${start_time}" ]; then
        get_memory_usage
        return 0
    fi
    
    local mem_avg=$(aws cloudwatch get-metric-statistics \
        --namespace "${CLOUDWATCH_NAMESPACE}" \
        --metric-name MemoryUtilization \
        --start-time "${start_time}" \
        --end-time "${end_time}" \
        --period 60 \
        --statistics Average \
        --region "${AWS_REGION}" \
        --query 'Datapoints[*].Average' \
        --output text 2>/dev/null | \
        awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "0"}')
    
    echo "${mem_avg:-0}"
}

get_request_rate() {
    # Get request rate from application metrics
    local period="${1:-300}"
    local end_time=$(date -u +%Y-%m-%dT%H:%M:%S)
    local start_time=$(date -u -d "${period} seconds ago" +%Y-%m-%dT%H:%M:%S 2>/dev/null || \
        date -u -v-${period}S +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "")
    
    if [ -z "${start_time}" ]; then
        echo "0"
        return 0
    fi
    
    local req_rate=$(aws cloudwatch get-metric-statistics \
        --namespace "${CLOUDWATCH_NAMESPACE}" \
        --metric-name RequestCount \
        --start-time "${start_time}" \
        --end-time "${end_time}" \
        --period 60 \
        --statistics Sum \
        --region "${AWS_REGION}" \
        --query 'Datapoints[*].Sum' \
        --output text 2>/dev/null | \
        awk '{sum+=$1} END {print sum/5}') # Average per minute
    
    echo "${req_rate:-0}"
}

check_cooldown() {
    local current_time=$(date +%s)
    local time_since_last_scale=$((current_time - LAST_SCALE_TIME))
    local required_cooldown=0
    
    if [ "${LAST_SCALE_ACTION}" = "scale_up" ]; then
        required_cooldown=${SCALE_UP_COOLDOWN}
    elif [ "${LAST_SCALE_ACTION}" = "scale_down" ]; then
        required_cooldown=${SCALE_DOWN_COOLDOWN}
    fi
    
    if [ $time_since_last_scale -lt $required_cooldown ]; then
        local remaining=$((required_cooldown - time_since_last_scale))
        log_info "In cooldown period. ${remaining}s remaining"
        return 1
    fi
    
    return 0
}

scale_up() {
    local current_capacity
    current_capacity=$(get_current_capacity)
    
    if [ "${current_capacity}" -ge "${MAX_INSTANCES}" ]; then
        log_warn "Already at maximum capacity: ${current_capacity}"
        return 1
    fi
    
    if ! check_cooldown; then
        return 1
    fi
    
    local new_capacity=$((current_capacity + 1))
    log_info "Scaling up: ${current_capacity} -> ${new_capacity}"
    
    if aws autoscaling set-desired-capacity \
        --auto-scaling-group-name "${AUTO_SCALING_GROUP}" \
        --desired-capacity "${new_capacity}" \
        --region "${AWS_REGION}" > /dev/null 2>&1; then
        
        LAST_SCALE_ACTION="scale_up"
        LAST_SCALE_TIME=$(date +%s)
        
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "ScaleUpAction" 1 "Count"
        send_sns_alert "${SNS_TOPIC_ARN:-}" \
            "Auto Scaling: Scale Up" \
            "Auto Scaling Group: ${AUTO_SCALING_GROUP}\nCapacity: ${current_capacity} -> ${new_capacity}"
        
        log_success "Scaled up successfully to ${new_capacity} instances"
        return 0
    else
        log_error "Failed to scale up"
        return 1
    fi
}

scale_down() {
    local current_capacity
    current_capacity=$(get_current_capacity)
    
    if [ "${current_capacity}" -le "${MIN_INSTANCES}" ]; then
        log_warn "Already at minimum capacity: ${current_capacity}"
        return 1
    fi
    
    if ! check_cooldown; then
        return 1
    fi
    
    local new_capacity=$((current_capacity - 1))
    log_info "Scaling down: ${current_capacity} -> ${new_capacity}"
    
    if aws autoscaling set-desired-capacity \
        --auto-scaling-group-name "${AUTO_SCALING_GROUP}" \
        --desired-capacity "${new_capacity}" \
        --region "${AWS_REGION}" > /dev/null 2>&1; then
        
        LAST_SCALE_ACTION="scale_down"
        LAST_SCALE_TIME=$(date +%s)
        
        send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "ScaleDownAction" 1 "Count"
        send_sns_alert "${SNS_TOPIC_ARN:-}" \
            "Auto Scaling: Scale Down" \
            "Auto Scaling Group: ${AUTO_SCALING_GROUP}\nCapacity: ${current_capacity} -> ${new_capacity}"
        
        log_success "Scaled down successfully to ${new_capacity} instances"
        return 0
    else
        log_error "Failed to scale down"
        return 1
    fi
}

evaluate_scaling() {
    log_info "Evaluating scaling conditions..."
    
    local cpu_avg
    cpu_avg=$(get_average_cpu 300)
    local mem_avg
    mem_avg=$(get_average_memory 300)
    local req_rate
    req_rate=$(get_request_rate 300)
    
    log_info "Current metrics:"
    log_info "  CPU: ${cpu_avg}%"
    log_info "  Memory: ${mem_avg}%"
    log_info "  Request Rate: ${req_rate} req/min"
    
    # Send metrics
    send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "AverageCPU" "${cpu_avg}" "Percent"
    send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "AverageMemory" "${mem_avg}" "Percent"
    send_cloudwatch_metric "${CLOUDWATCH_NAMESPACE}" "RequestRate" "${req_rate}" "Count"
    
    # Check scale up conditions
    if (( $(echo "${cpu_avg} > ${CPU_SCALE_UP_THRESHOLD}" | bc -l) )) || \
       (( $(echo "${mem_avg} > ${MEMORY_SCALE_UP_THRESHOLD}" | bc -l) )) || \
       (( $(echo "${req_rate} > ${REQUEST_RATE_SCALE_UP}" | bc -l) )); then
        log_info "Scale up conditions met"
        scale_up
        return 0
    fi
    
    # Check scale down conditions
    if (( $(echo "${cpu_avg} < ${CPU_SCALE_DOWN_THRESHOLD}" | bc -l) )) && \
       (( $(echo "${mem_avg} < ${MEMORY_SCALE_DOWN_THRESHOLD}" | bc -l) )) && \
       (( $(echo "${req_rate} < ${REQUEST_RATE_SCALE_DOWN}" | bc -l) )); then
        log_info "Scale down conditions met"
        scale_down
        return 0
    fi
    
    log_info "No scaling action needed"
    return 0
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "=========================================="
    log_info "Starting automated scaling evaluation"
    log_info "=========================================="
    
    if [ -z "${AUTO_SCALING_GROUP}" ]; then
        log_error "AUTO_SCALING_GROUP not specified"
        exit 1
    fi
    
    if ! check_aws_credentials; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    evaluate_scaling
    
    log_info "=========================================="
    log_success "Scaling evaluation completed"
    log_info "=========================================="
}

# Run main function
main "$@"

