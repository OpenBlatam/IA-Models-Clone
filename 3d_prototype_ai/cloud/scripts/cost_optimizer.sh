#!/bin/bash
# Cost optimization script
# Analyzes and optimizes AWS costs

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Default values
readonly AWS_REGION="${AWS_REGION:-us-east-1}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-./cost_reports}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Analyze and optimize AWS costs.

COMMANDS:
    analyze              Analyze current costs
    recommendations       Get cost optimization recommendations
    unused-resources      Find unused resources
    rightsize            Get rightsizing recommendations
    savings-plans        Analyze Savings Plans opportunities
    report               Generate cost report

OPTIONS:
    -r, --region REGION   AWS region (default: us-east-1)
    -o, --output DIR      Output directory (default: ./cost_reports)
    -d, --days DAYS       Number of days to analyze (default: 30)
    -h, --help            Show this help message

EXAMPLES:
    $0 analyze --days 7
    $0 recommendations
    $0 unused-resources
    $0 report --output ./reports

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    DAYS=30
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -d|--days)
                DAYS="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            analyze|recommendations|unused-resources|rightsize|savings-plans|report)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Analyze costs
analyze_costs() {
    local days="${1}"
    local output_dir="${2}"
    
    log_info "Analyzing AWS costs for last ${days} days..."
    
    mkdir -p "${output_dir}"
    local report_file="${output_dir}/cost_analysis_$(date +%Y%m%d_%H%M%S).json"
    
    # Get cost and usage data
    aws ce get-cost-and-usage \
        --time-period Start=$(date -u -d "${days} days ago" +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
        --granularity MONTHLY \
        --metrics BlendedCost \
        --group-by Type=SERVICE \
        --region "${AWS_REGION}" \
        > "${report_file}" 2>/dev/null || log_warn "Cost Explorer not available or no permissions"
    
    # Get EC2 costs
    local ec2_costs
    ec2_costs=$(aws ce get-cost-and-usage \
        --time-period Start=$(date -u -d "${days} days ago" +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
        --granularity MONTHLY \
        --metrics BlendedCost \
        --filter file://<(echo '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Compute Cloud - Compute"]}}') \
        --region "${AWS_REGION}" \
        --query 'ResultsByTime[0].Total.BlendedCost.Amount' \
        --output text 2>/dev/null || echo "N/A")
    
    log_info "EC2 Costs (last ${days} days): \$${ec2_costs}"
    log_info "Full report saved to: ${report_file}"
}

# Get recommendations
get_recommendations() {
    local output_dir="${1}"
    
    log_info "Getting cost optimization recommendations..."
    
    mkdir -p "${output_dir}"
    local report_file="${output_dir}/recommendations_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "${report_file}" << EOF
Cost Optimization Recommendations
==================================

1. Instance Rightsizing
   - Review EC2 instance utilization
   - Consider downsizing underutilized instances
   - Use CloudWatch metrics to identify opportunities

2. Reserved Instances
   - Purchase Reserved Instances for predictable workloads
   - Consider Savings Plans for flexibility
   - Analyze usage patterns before committing

3. Spot Instances
   - Use Spot Instances for fault-tolerant workloads
   - Can save up to 90% compared to On-Demand

4. Auto Scaling
   - Implement auto-scaling to match demand
   - Scale down during off-peak hours
   - Use scheduled scaling for predictable patterns

5. Storage Optimization
   - Review EBS volumes for unused space
   - Delete unattached volumes
   - Use appropriate volume types

6. Data Transfer
   - Minimize data transfer costs
   - Use CloudFront for content delivery
   - Optimize S3 bucket locations

7. Monitoring
   - Set up billing alerts
   - Monitor cost trends
   - Review monthly cost reports

EOF
    
    log_info "Recommendations saved to: ${report_file}"
    cat "${report_file}"
}

# Find unused resources
find_unused_resources() {
    local output_dir="${1}"
    
    log_info "Finding unused resources..."
    
    mkdir -p "${output_dir}"
    local report_file="${output_dir}/unused_resources_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Unused Resources Report"
        echo "======================"
        echo ""
        
        # Unattached EBS volumes
        echo "Unattached EBS Volumes:"
        aws ec2 describe-volumes \
            --filters Name=status,Values=available \
            --query 'Volumes[*].[VolumeId,Size,CreateTime]' \
            --output table \
            --region "${AWS_REGION}" 2>/dev/null || echo "  None found or no permissions"
        echo ""
        
        # Unused Elastic IPs
        echo "Unused Elastic IPs:"
        aws ec2 describe-addresses \
            --query 'Addresses[?AssociationId==null].[PublicIp,AllocationId]' \
            --output table \
            --region "${AWS_REGION}" 2>/dev/null || echo "  None found or no permissions"
        echo ""
        
        # Stopped instances
        echo "Stopped EC2 Instances:"
        aws ec2 describe-instances \
            --filters Name=instance-state-name,Values=stopped \
            --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime]' \
            --output table \
            --region "${AWS_REGION}" 2>/dev/null || echo "  None found or no permissions"
        
    } > "${report_file}"
    
    log_info "Unused resources report saved to: ${report_file}"
    cat "${report_file}"
}

# Rightsizing recommendations
get_rightsizing() {
    local output_dir="${1}"
    
    log_info "Getting rightsizing recommendations..."
    
    mkdir -p "${output_dir}"
    local report_file="${output_dir}/rightsizing_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "${report_file}" << EOF
Rightsizing Recommendations
==========================

To get rightsizing recommendations:

1. Enable Cost Explorer Rightsizing Recommendations
2. Review CloudWatch metrics for CPU and Memory utilization
3. Consider:
   - Downsize if consistently < 40% utilization
   - Upsize if consistently > 80% utilization
   - Use burstable instances (t3) for variable workloads

Current Instance Analysis:
EOF
    
    # Get instance utilization
    aws ec2 describe-instances \
        --filters Name=instance-state-name,Values=running \
        --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime]' \
        --output table \
        --region "${AWS_REGION}" 2>/dev/null >> "${report_file}" || echo "  Could not retrieve instance data" >> "${report_file}"
    
    log_info "Rightsizing report saved to: ${report_file}"
    cat "${report_file}"
}

# Generate cost report
generate_report() {
    local days="${1}"
    local output_dir="${2}"
    
    log_info "Generating comprehensive cost report..."
    
    mkdir -p "${output_dir}"
    local report_file="${output_dir}/cost_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "AWS Cost Report"
        echo "==============="
        echo "Generated: $(date)"
        echo "Period: Last ${days} days"
        echo "Region: ${AWS_REGION}"
        echo ""
        
        # Service costs
        echo "Service Costs:"
        aws ce get-cost-and-usage \
            --time-period Start=$(date -u -d "${days} days ago" +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
            --granularity MONTHLY \
            --metrics BlendedCost \
            --group-by Type=SERVICE \
            --region "${AWS_REGION}" \
            --query 'ResultsByTime[0].Groups[*].[Keys[0],Metrics.BlendedCost.Amount]' \
            --output table 2>/dev/null || echo "  Cost Explorer data not available"
        echo ""
        
        # Instance summary
        echo "EC2 Instance Summary:"
        aws ec2 describe-instances \
            --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name]' \
            --output table \
            --region "${AWS_REGION}" 2>/dev/null || echo "  Could not retrieve instance data"
        
    } > "${report_file}"
    
    log_info "Cost report saved to: ${report_file}"
    cat "${report_file}"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        analyze)
            analyze_costs "${DAYS}" "${OUTPUT_DIR}"
            ;;
        recommendations)
            get_recommendations "${OUTPUT_DIR}"
            ;;
        unused-resources)
            find_unused_resources "${OUTPUT_DIR}"
            ;;
        rightsize)
            get_rightsizing "${OUTPUT_DIR}"
            ;;
        savings-plans)
            log_info "Savings Plans analysis - Use AWS Cost Explorer console for detailed analysis"
            ;;
        report)
            generate_report "${DAYS}" "${OUTPUT_DIR}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


