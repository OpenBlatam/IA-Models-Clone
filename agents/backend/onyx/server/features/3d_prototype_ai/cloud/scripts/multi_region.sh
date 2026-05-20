#!/bin/bash
# Multi-region deployment script
# Manages deployments across multiple AWS regions

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly REGIONS="${REGIONS:-us-east-1,us-west-2,eu-west-1}"
readonly PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Manage multi-region deployments.

COMMANDS:
    deploy REGION        Deploy to specific region
    deploy-all           Deploy to all regions
    status               Show status across regions
    sync                 Sync data across regions
    failover             Failover to another region

OPTIONS:
    -r, --regions REGIONS    Comma-separated regions (default: us-east-1,us-west-2,eu-west-1)
    -p, --primary REGION     Primary region (default: us-east-1)
    -h, --help               Show this help message

EXAMPLES:
    $0 deploy us-west-2
    $0 deploy-all
    $0 status
    $0 failover us-west-2

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    TARGET_REGION=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--regions)
                REGIONS="$2"
                shift 2
                ;;
            -p|--primary)
                PRIMARY_REGION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            deploy|deploy-all|status|sync|failover)
                COMMAND="$1"
                if [ "$COMMAND" = "deploy" ] || [ "$COMMAND" = "failover" ]; then
                    TARGET_REGION="$2"
                    shift 2
                else
                    shift
                fi
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

# Deploy to region
deploy_to_region() {
    local region="${1}"
    
    log_info "Deploying to region: ${region}"
    
    # Set AWS region
    export AWS_REGION="${region}"
    
    # Run deployment
    ./scripts/deploy.sh --region "${region}" || {
        log_error "Deployment to ${region} failed"
        return 1
    }
    
    log_info "Deployment to ${region} completed"
}

# Deploy to all regions
deploy_all_regions() {
    log_info "Deploying to all regions: ${REGIONS}"
    
    local failed_regions=()
    local success_regions=()
    
    IFS=',' read -ra REGION_ARRAY <<< "${REGIONS}"
    for region in "${REGION_ARRAY[@]}"; do
        region=$(echo "${region}" | xargs)  # Trim whitespace
        if deploy_to_region "${region}"; then
            success_regions+=("${region}")
        else
            failed_regions+=("${region}")
        fi
    done
    
    echo ""
    log_info "Deployment Summary:"
    log_info "  Successful: ${success_regions[*]}"
    if [ ${#failed_regions[@]} -gt 0 ]; then
        log_error "  Failed: ${failed_regions[*]}"
        return 1
    fi
}

# Show status across regions
show_status() {
    log_info "Status across regions:"
    echo ""
    
    IFS=',' read -ra REGION_ARRAY <<< "${REGIONS}"
    for region in "${REGION_ARRAY[@]}"; do
        region=$(echo "${region}" | xargs)
        log_info "Region: ${region}"
        
        # Get instances in region
        local instances
        instances=$(aws ec2 describe-instances \
            --filters "Name=tag:Project,Values=3D-Prototype-AI" \
                     "Name=instance-state-name,Values=running" \
            --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress,State.Name]' \
            --output text \
            --region "${region}" 2>/dev/null || echo "")
        
        if [ -n "${instances}" ]; then
            echo "${instances}" | while read -r instance_id ip state; do
                printf "  Instance: %s | IP: %s | State: %s\n" "${instance_id}" "${ip}" "${state}"
            done
        else
            echo "  No instances found"
        fi
        echo ""
    done
}

# Sync data across regions
sync_data() {
    log_info "Syncing data across regions..."
    
    # Get primary region instance
    local primary_ip
    primary_ip=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=3D-Prototype-AI" \
                 "Name=tag:Region,Values=${PRIMARY_REGION}" \
                 "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region "${PRIMARY_REGION}" 2>/dev/null || echo "")
    
    if [ -z "${primary_ip}" ] || [ "${primary_ip}" = "None" ]; then
        log_error "Primary region instance not found"
        return 1
    fi
    
    log_info "Syncing from primary region (${PRIMARY_REGION})..."
    
    # Sync to other regions
    IFS=',' read -ra REGION_ARRAY <<< "${REGIONS}"
    for region in "${REGION_ARRAY[@]}"; do
        region=$(echo "${region}" | xargs)
        if [ "${region}" != "${PRIMARY_REGION}" ]; then
            log_info "Syncing to ${region}..."
            # Implement sync logic here
        fi
    done
}

# Failover to region
failover_to_region() {
    local target_region="${1}"
    
    log_warn "Failover to region: ${target_region}"
    log_warn "This will switch traffic from ${PRIMARY_REGION} to ${target_region}"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Failover cancelled"
        return 0
    fi
    
    log_info "Initiating failover to ${target_region}..."
    
    # Update DNS/load balancer to point to target region
    # This is a placeholder - implement based on your infrastructure
    
    log_info "Failover to ${target_region} completed"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        deploy)
            if [ -z "${TARGET_REGION}" ]; then
                error_exit 1 "Target region is required"
            fi
            deploy_to_region "${TARGET_REGION}"
            ;;
        deploy-all)
            deploy_all_regions
            ;;
        status)
            show_status
            ;;
        sync)
            sync_data
            ;;
        failover)
            if [ -z "${TARGET_REGION}" ]; then
                error_exit 1 "Target region is required"
            fi
            failover_to_region "${TARGET_REGION}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


