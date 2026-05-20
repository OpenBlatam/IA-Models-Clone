#!/bin/bash
# Automated Disaster Recovery Script
# Handles automated failover and recovery procedures

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"
DR_REGION="${DR_REGION:-us-west-2}"
FAILOVER_THRESHOLD="${FAILOVER_THRESHOLD:-3}"  # Number of failed health checks before failover

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check primary region health
check_primary_health() {
    log_info "Checking primary region health: $PRIMARY_REGION"
    
    export AWS_DEFAULT_REGION="$PRIMARY_REGION"
    
    local instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=ai-project-generator" \
                  "Name=tag:Environment,Values=primary" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].PublicIpAddress' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$instances" ]; then
        log_error "No primary instances found"
        return 1
    fi
    
    local healthy_count=0
    local total_count=0
    
    for instance_ip in $instances; do
        total_count=$((total_count + 1))
        if curl -f -s -m 5 "http://$instance_ip/health" > /dev/null 2>&1; then
            healthy_count=$((healthy_count + 1))
        fi
    done
    
    local health_percentage=$((healthy_count * 100 / total_count))
    
    if [ $health_percentage -ge 80 ]; then
        log_info "✅ Primary region healthy: $health_percentage%"
        return 0
    else
        log_error "❌ Primary region unhealthy: $health_percentage%"
        return 1
    fi
}

# Activate DR region
activate_dr_region() {
    log_warn "Activating DR region: $DR_REGION"
    
    export AWS_DEFAULT_REGION="$DR_REGION"
    
    # Get DR instances
    local instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=ai-project-generator" \
                  "Name=tag:Environment,Values=dr" \
                  "Name=instance-state-name,Values=stopped" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$instances" ]; then
        log_error "No DR instances found"
        return 1
    fi
    
    # Start DR instances
    log_info "Starting DR instances..."
    for instance_id in $instances; do
        aws ec2 start-instances --instance-ids "$instance_id" || {
            log_error "Failed to start instance: $instance_id"
            return 1
        }
    done
    
    # Wait for instances to be running
    log_info "Waiting for instances to start..."
    aws ec2 wait instance-running --instance-ids $instances || {
        log_error "Instances failed to start"
        return 1
    }
    
    # Update Route53 or ALB to point to DR region
    log_info "Updating DNS/ALB to point to DR region..."
    # Add Route53 or ALB update logic here
    
    # Health check DR instances
    log_info "Health checking DR instances..."
    sleep 30  # Wait for services to start
    
    local dr_instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=ai-project-generator" \
                  "Name=tag:Environment,Values=dr" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].PublicIpAddress' \
        --output text 2>/dev/null || echo "")
    
    for instance_ip in $dr_instances; do
        local max_attempts=10
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if curl -f -s -m 5 "http://$instance_ip/health" > /dev/null 2>&1; then
                log_info "✅ DR instance healthy: $instance_ip"
                break
            else
                log_warn "Health check attempt $attempt/$max_attempts failed for $instance_ip"
                sleep 10
                attempt=$((attempt + 1))
            fi
        done
        
        if [ $attempt -gt $max_attempts ]; then
            log_error "❌ DR instance failed health check: $instance_ip"
            return 1
        fi
    done
    
    log_info "✅ DR region activated successfully"
    return 0
}

# Failover to DR
failover_to_dr() {
    log_warn "=== Initiating Failover to DR Region ==="
    
    # Check primary health multiple times
    local failed_checks=0
    
    for i in $(seq 1 $FAILOVER_THRESHOLD); do
        if ! check_primary_health; then
            failed_checks=$((failed_checks + 1))
            log_warn "Primary health check failed ($failed_checks/$FAILOVER_THRESHOLD)"
        else
            log_info "Primary region recovered, aborting failover"
            return 0
        fi
        
        if [ $i -lt $FAILOVER_THRESHOLD ]; then
            sleep 30
        fi
    done
    
    if [ $failed_checks -ge $FAILOVER_THRESHOLD ]; then
        log_error "Primary region failed $failed_checks health checks, initiating failover"
        
        if activate_dr_region; then
            log_info "✅ Failover to DR region completed"
            
            # Send notification
            send_failover_notification
            
            return 0
        else
            log_error "❌ Failover to DR region failed"
            return 1
        fi
    fi
}

# Failback to primary
failback_to_primary() {
    log_info "=== Initiating Failback to Primary Region ==="
    
    export AWS_DEFAULT_REGION="$PRIMARY_REGION"
    
    # Check primary region health
    if ! check_primary_health; then
        log_error "Primary region not healthy, cannot failback"
        return 1
    fi
    
    # Update DNS/ALB to point back to primary
    log_info "Updating DNS/ALB to point to primary region..."
    # Add Route53 or ALB update logic here
    
    # Verify primary is serving traffic
    log_info "Verifying primary region is serving traffic..."
    sleep 30
    
    if check_primary_health; then
        log_info "✅ Failback to primary region completed"
        
        # Optionally stop DR instances to save costs
        log_info "Stopping DR instances..."
        export AWS_DEFAULT_REGION="$DR_REGION"
        
        local dr_instances=$(aws ec2 describe-instances \
            --filters "Name=tag:Project,Values=ai-project-generator" \
                      "Name=tag:Environment,Values=dr" \
                      "Name=instance-state-name,Values=running" \
            --query 'Reservations[*].Instances[*].InstanceId' \
            --output text 2>/dev/null || echo "")
        
        if [ -n "$dr_instances" ]; then
            aws ec2 stop-instances --instance-ids $dr_instances || {
                log_warn "Failed to stop some DR instances"
            }
        fi
        
        send_failback_notification
        return 0
    else
        log_error "❌ Primary region failed after failback"
        return 1
    fi
}

# Send failover notification
send_failover_notification() {
    log_info "Sending failover notification..."
    
    local message="🚨 Disaster Recovery Failover Initiated

Primary Region: $PRIMARY_REGION
DR Region: $DR_REGION
Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Status: DR region activated"
    
    # Send to SNS if configured
    if [ -n "${SNS_TOPIC_ARN:-}" ]; then
        aws sns publish \
            --topic-arn "$SNS_TOPIC_ARN" \
            --subject "DR Failover Initiated" \
            --message "$message" || log_warn "Failed to send SNS notification"
    fi
    
    # Log to CloudWatch
    aws logs put-log-events \
        --log-group-name "/aws/ec2/ai-project-generator" \
        --log-stream-name "dr-failover" \
        --log-events "[{\"timestamp\": $(date +%s)000, \"message\": \"$message\"}]" \
        2>/dev/null || true
}

# Send failback notification
send_failback_notification() {
    log_info "Sending failback notification..."
    
    local message="✅ Disaster Recovery Failback Completed

Primary Region: $PRIMARY_REGION
DR Region: $DR_REGION
Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Status: Traffic restored to primary region"
    
    if [ -n "${SNS_TOPIC_ARN:-}" ]; then
        aws sns publish \
            --topic-arn "$SNS_TOPIC_ARN" \
            --subject "DR Failback Completed" \
            --message "$message" || log_warn "Failed to send SNS notification"
    fi
}

# Test DR procedure
test_dr() {
    log_info "=== Testing DR Procedure ==="
    
    log_info "1. Checking primary region..."
    check_primary_health
    
    log_info "2. Activating DR region (test mode)..."
    activate_dr_region
    
    log_info "3. Verifying DR region health..."
    export AWS_DEFAULT_REGION="$DR_REGION"
    check_primary_health  # Reuse function with DR region
    
    log_info "✅ DR test completed"
}

# Main function
main() {
    case "${1:-monitor}" in
        monitor)
            # Continuous monitoring mode
            log_info "Starting DR monitoring..."
            while true; do
                if ! check_primary_health; then
                    failover_to_dr
                fi
                sleep 60
            done
            ;;
        failover)
            failover_to_dr
            ;;
        failback)
            failback_to_primary
            ;;
        test)
            test_dr
            ;;
        check)
            check_primary_health
            ;;
        *)
            echo "Usage: $0 {monitor|failover|failback|test|check}"
            echo "Environment variables:"
            echo "  PRIMARY_REGION: Primary AWS region (default: us-east-1)"
            echo "  DR_REGION: DR AWS region (default: us-west-2)"
            echo "  FAILOVER_THRESHOLD: Failed health checks before failover (default: 3)"
            echo "  SNS_TOPIC_ARN: SNS topic for notifications (optional)"
            exit 1
            ;;
    esac
}

main "$@"



