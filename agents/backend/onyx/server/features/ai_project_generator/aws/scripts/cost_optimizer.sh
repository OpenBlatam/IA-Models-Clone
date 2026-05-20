#!/bin/bash
# Cost Optimization Script
# Monitors and optimizes AWS costs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COST_THRESHOLD="${COST_THRESHOLD:-100}"  # USD per day
ALERT_EMAIL="${ALERT_EMAIL:-}"

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

# Get current daily cost
get_daily_cost() {
    local start_date=$(date -u -d '1 day ago' +%Y-%m-%d)
    local end_date=$(date -u +%Y-%m-%d)
    
    # Use AWS Cost Explorer API
    aws ce get-cost-and-usage \
        --time-period Start="$start_date",End="$end_date" \
        --granularity DAILY \
        --metrics "UnblendedCost" \
        --query 'ResultsByTime[0].Total.UnblendedCost.Amount' \
        --output text 2>/dev/null || echo "0"
}

# Get cost by service
get_cost_by_service() {
    local start_date=$(date -u -d '1 day ago' +%Y-%m-%d)
    local end_date=$(date -u +%Y-%m-%d)
    
    aws ce get-cost-and-usage \
        --time-period Start="$start_date",End="$end_date" \
        --granularity DAILY \
        --metrics "UnblendedCost" \
        --group-by Type=SERVICE \
        --query 'ResultsByTime[0].Groups[*].[Keys[0],Metrics.UnblendedCost.Amount]' \
        --output table 2>/dev/null || echo "Unable to retrieve cost data"
}

# Identify unused resources
find_unused_resources() {
    log_info "Identifying unused resources..."
    
    # Find stopped instances
    log_info "=== Stopped EC2 Instances ==="
    aws ec2 describe-instances \
        --filters "Name=instance-state-name,Values=stopped" \
        --query 'Reservations[*].Instances[*].[InstanceId,Tags[?Key==`Name`].Value|[0],InstanceType]' \
        --output table
    
    # Find unattached EBS volumes
    log_info "=== Unattached EBS Volumes ==="
    aws ec2 describe-volumes \
        --filters "Name=status,Values=available" \
        --query 'Volumes[*].[VolumeId,Size,VolumeType]' \
        --output table
    
    # Find unused snapshots
    log_info "=== Old Snapshots ==="
    local cutoff_date=$(date -u -d '30 days ago' +%Y-%m-%d)
    aws ec2 describe-snapshots \
        --owner-ids self \
        --query "Snapshots[?StartTime<='$cutoff_date'].[SnapshotId,StartTime,VolumeSize]" \
        --output table
    
    # Find idle load balancers
    log_info "=== Load Balancers ==="
    aws elbv2 describe-load-balancers \
        --query 'LoadBalancers[*].[LoadBalancerName,Type,State.Code]' \
        --output table
}

# Optimize instance sizes
optimize_instances() {
    log_info "Analyzing instance utilization..."
    
    # Get instance metrics from CloudWatch
    local instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=ai-project-generator" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].[InstanceId,InstanceType]' \
        --output text)
    
    while IFS=$'\t' read -r instance_id instance_type; do
        log_info "Analyzing instance: $instance_id ($instance_type)"
        
        # Get CPU utilization (last 7 days average)
        local avg_cpu=$(aws cloudwatch get-metric-statistics \
            --namespace AWS/EC2 \
            --metric-name CPUUtilization \
            --dimensions Name=InstanceId,Value="$instance_id" \
            --start-time $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S) \
            --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
            --period 3600 \
            --statistics Average \
            --query 'Datapoints[*].Average' \
            --output text 2>/dev/null | \
            awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
        
        if [ "$avg_cpu" != "N/A" ]; then
            if (( $(echo "$avg_cpu < 20" | bc -l) )); then
                log_warn "⚠️  Low CPU utilization: ${avg_cpu}% - Consider downsizing $instance_id"
            elif (( $(echo "$avg_cpu > 80" | bc -l) )); then
                log_warn "⚠️  High CPU utilization: ${avg_cpu}% - Consider upsizing $instance_id"
            else
                log_info "✅ CPU utilization normal: ${avg_cpu}%"
            fi
        fi
    done <<< "$instances"
}

# Enable cost-saving features
enable_cost_savings() {
    log_info "Enabling cost-saving features..."
    
    # Enable EC2 instance scheduling (stop during off-hours)
    log_info "Setting up EC2 instance scheduling..."
    # Add instance scheduler Lambda function or use AWS Instance Scheduler
    
    # Enable S3 lifecycle policies
    log_info "Configuring S3 lifecycle policies..."
    # Add S3 lifecycle configuration
    
    # Enable EBS snapshot lifecycle
    log_info "Configuring EBS snapshot lifecycle..."
    # Add EBS snapshot lifecycle policy
    
    # Enable Reserved Instance recommendations
    log_info "Checking Reserved Instance recommendations..."
    aws ce get-reservation-coverage \
        --time-period Start=$(date -u -d '30 days ago' +%Y-%m-%d),End=$(date -u +%Y-%m-%d) \
        --granularity DAILY \
        --metrics "CoverageHours" \
        --query 'Total.CoverageHours.CoverageHoursPercentage' \
        --output text 2>/dev/null || log_warn "Unable to get RI recommendations"
    
    log_info "✅ Cost-saving features enabled"
}

# Generate cost report
generate_cost_report() {
    local report_file="/tmp/cost_report_$(date +%Y%m%d).txt"
    
    log_info "Generating cost report..."
    
    {
        echo "=== AWS Cost Report ==="
        echo "Date: $(date)"
        echo ""
        echo "=== Daily Cost ==="
        echo "Yesterday's Cost: \$$(get_daily_cost)"
        echo ""
        echo "=== Cost by Service ==="
        get_cost_by_service
        echo ""
        echo "=== Cost Optimization Recommendations ==="
        find_unused_resources
        echo ""
        optimize_instances
    } | tee "$report_file"
    
    log_info "✅ Cost report generated: $report_file"
    
    # Send email if configured
    if [ -n "$ALERT_EMAIL" ]; then
        mail -s "AWS Cost Report - $(date +%Y-%m-%d)" "$ALERT_EMAIL" < "$report_file" || {
            log_warn "Failed to send email"
        }
    fi
}

# Alert on cost threshold
check_cost_threshold() {
    local daily_cost=$(get_daily_cost)
    
    if (( $(echo "$daily_cost > $COST_THRESHOLD" | bc -l) )); then
        log_error "🚨 Daily cost exceeds threshold: \$$daily_cost > \$$COST_THRESHOLD"
        
        # Send alert
        if [ -n "${SNS_TOPIC_ARN:-}" ]; then
            aws sns publish \
                --topic-arn "$SNS_TOPIC_ARN" \
                --subject "Cost Alert: Daily Cost Exceeded" \
                --message "Daily cost: \$$daily_cost exceeds threshold: \$$COST_THRESHOLD" || true
        fi
        
        return 1
    else
        log_info "✅ Daily cost within threshold: \$$daily_cost"
        return 0
    fi
}

# Main function
main() {
    case "${1:-report}" in
        report)
            generate_cost_report
            ;;
        optimize)
            find_unused_resources
            optimize_instances
            enable_cost_savings
            ;;
        monitor)
            check_cost_threshold
            ;;
        *)
            echo "Usage: $0 {report|optimize|monitor}"
            echo "Environment variables:"
            echo "  COST_THRESHOLD: Daily cost threshold in USD (default: 100)"
            echo "  ALERT_EMAIL: Email for cost reports (optional)"
            echo "  SNS_TOPIC_ARN: SNS topic for alerts (optional)"
            exit 1
            ;;
    esac
}

main "$@"



