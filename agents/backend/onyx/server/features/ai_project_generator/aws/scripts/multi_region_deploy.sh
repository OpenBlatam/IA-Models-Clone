#!/bin/bash
# Multi-Region Deployment Script
# Deploys application to multiple AWS regions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGIONS="${REGIONS:-us-east-1,us-west-2,eu-west-1}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-parallel}"  # parallel or sequential

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

# Deploy to single region
deploy_to_region() {
    local region="$1"
    
    log_info "Deploying to region: $region"
    
    # Set AWS region
    export AWS_DEFAULT_REGION="$region"
    
    # Get instances in region
    local instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=ai-project-generator" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].PublicIpAddress' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$instances" ]; then
        log_warn "No instances found in region $region"
        return 1
    fi
    
    # Deploy to each instance
    for instance_ip in $instances; do
        log_info "Deploying to instance: $instance_ip"
        
        # Use Ansible or SSH to deploy
        if command -v ansible-playbook > /dev/null 2>&1; then
            ansible-playbook -i "$instance_ip," \
                -e "ansible_user=ubuntu" \
                -e "ansible_ssh_private_key_file=~/.ssh/deploy_key" \
                playbooks/update.yml || {
                log_error "Deployment failed for $instance_ip"
                return 1
            }
        else
            log_warn "Ansible not available, skipping $instance_ip"
        fi
    done
    
    log_info "✅ Deployment completed for region: $region"
    return 0
}

# Health check for region
health_check_region() {
    local region="$1"
    
    log_info "Health checking region: $region"
    
    export AWS_DEFAULT_REGION="$region"
    
    local instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=ai-project-generator" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].PublicIpAddress' \
        --output text 2>/dev/null || echo "")
    
    local failed=0
    
    for instance_ip in $instances; do
        if curl -f -s -m 5 "http://$instance_ip/health" > /dev/null 2>&1; then
            log_info "✅ $instance_ip is healthy"
        else
            log_error "❌ $instance_ip health check failed"
            failed=$((failed + 1))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        log_info "✅ All instances healthy in region: $region"
        return 0
    else
        log_error "❌ $failed instances failed health check in region: $region"
        return 1
    fi
}

# Deploy to all regions
deploy_all_regions() {
    log_info "=== Multi-Region Deployment ==="
    log_info "Regions: $REGIONS"
    log_info "Strategy: $DEPLOYMENT_STRATEGY"
    
    IFS=',' read -ra REGION_ARRAY <<< "$REGIONS"
    local failed_regions=()
    
    if [ "$DEPLOYMENT_STRATEGY" == "parallel" ]; then
        log_info "Deploying to all regions in parallel..."
        
        # Deploy in parallel using background processes
        local pids=()
        for region in "${REGION_ARRAY[@]}"; do
            deploy_to_region "$region" &
            pids+=($!)
        done
        
        # Wait for all deployments
        local failed=0
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                log_error "Deployment failed for region: ${REGION_ARRAY[$i]}"
                failed_regions+=("${REGION_ARRAY[$i]}")
                failed=1
            fi
        done
        
        if [ $failed -eq 0 ]; then
            log_info "✅ All regions deployed successfully"
        else
            log_error "❌ Some regions failed deployment"
            return 1
        fi
    else
        log_info "Deploying to regions sequentially..."
        
        for region in "${REGION_ARRAY[@]}"; do
            if ! deploy_to_region "$region"; then
                failed_regions+=("$region")
            fi
        done
        
        if [ ${#failed_regions[@]} -eq 0 ]; then
            log_info "✅ All regions deployed successfully"
        else
            log_error "❌ Failed regions: ${failed_regions[*]}"
            return 1
        fi
    fi
    
    # Health check all regions
    log_info "Performing health checks..."
    local health_failed=0
    
    for region in "${REGION_ARRAY[@]}"; do
        if ! health_check_region "$region"; then
            health_failed=1
        fi
    done
    
    if [ $health_failed -eq 0 ]; then
        log_info "✅ All regions healthy"
        return 0
    else
        log_error "❌ Some regions failed health check"
        return 1
    fi
}

# Rollback region
rollback_region() {
    local region="$1"
    
    log_warn "Rolling back region: $region"
    
    export AWS_DEFAULT_REGION="$region"
    
    local instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=ai-project-generator" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].PublicIpAddress' \
        --output text 2>/dev/null || echo "")
    
    for instance_ip in $instances; do
        log_info "Rolling back instance: $instance_ip"
        ssh -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
            "sudo /opt/ai-project-generator/scripts/rollback.sh" || {
            log_error "Rollback failed for $instance_ip"
        }
    done
    
    log_info "✅ Rollback completed for region: $region"
}

# Main function
main() {
    case "${1:-deploy}" in
        deploy)
            deploy_all_regions
            ;;
        health-check)
            IFS=',' read -ra REGION_ARRAY <<< "$REGIONS"
            for region in "${REGION_ARRAY[@]}"; do
                health_check_region "$region"
            done
            ;;
        rollback)
            local region="${2:-}"
            if [ -z "$region" ]; then
                IFS=',' read -ra REGION_ARRAY <<< "$REGIONS"
                for r in "${REGION_ARRAY[@]}"; do
                    rollback_region "$r"
                done
            else
                rollback_region "$region"
            fi
            ;;
        *)
            echo "Usage: $0 {deploy|health-check|rollback} [region]"
            echo "Environment variables:"
            echo "  REGIONS: Comma-separated list of regions (default: us-east-1,us-west-2,eu-west-1)"
            echo "  DEPLOYMENT_STRATEGY: parallel or sequential (default: parallel)"
            exit 1
            ;;
    esac
}

main "$@"



