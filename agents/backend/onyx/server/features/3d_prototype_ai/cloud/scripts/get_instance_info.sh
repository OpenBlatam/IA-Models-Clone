#!/bin/bash
# Get EC2 instance information
# Useful for CI/CD and automation scripts

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly AWS_REGION="${AWS_REGION:-us-east-1}"
readonly PROJECT_NAME="${PROJECT_NAME:-3d-prototype-ai}"
readonly ENVIRONMENT="${ENVIRONMENT:-prod}"

# Function to get instance from Terraform
get_instance_from_terraform() {
    local terraform_dir="${CLOUD_DIR}/terraform"
    
    if [ ! -d "${terraform_dir}" ]; then
        return 1
    fi
    
    cd "${terraform_dir}"
    
    if [ -f "terraform.tfstate" ]; then
        local instance_id
        instance_id=$(terraform output -raw instance_id 2>/dev/null || echo "")
        local instance_ip
        instance_ip=$(terraform output -raw instance_public_ip 2>/dev/null || echo "")
        
        if [ -n "${instance_id}" ] && [ "${instance_id}" != "null" ]; then
            echo "${instance_id}|${instance_ip}"
            return 0
        fi
    fi
    
    cd - > /dev/null
    return 1
}

# Function to get instance from AWS by tags
get_instance_from_aws() {
    if ! command -v aws &> /dev/null; then
        return 1
    fi
    
    if ! check_aws_credentials; then
        return 1
    fi
    
    local instance_info
    instance_info=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=${PROJECT_NAME}" \
                  "Name=tag:Environment,Values=${ENVIRONMENT}" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
        --output text \
        --region "${AWS_REGION}" 2>/dev/null || echo "")
    
    if [ -n "${instance_info}" ] && [ "${instance_info}" != "None None" ]; then
        local instance_id
        instance_id=$(echo "${instance_info}" | awk '{print $1}')
        local instance_ip
        instance_ip=$(echo "${instance_info}" | awk '{print $2}')
        
        if [ -n "${instance_id}" ] && [ "${instance_id}" != "None" ]; then
            echo "${instance_id}|${instance_ip}"
            return 0
        fi
    fi
    
    return 1
}

# Main function
main() {
    log_info "Getting EC2 instance information..."
    
    local instance_info=""
    
    # Try Terraform first
    if instance_info=$(get_instance_from_terraform); then
        log_info "Found instance from Terraform state"
    # Try AWS by tags
    elif instance_info=$(get_instance_from_aws); then
        log_info "Found instance from AWS tags"
    else
        log_error "Could not find EC2 instance"
        log_info "Make sure:"
        log_info "  1. Instance is running"
        log_info "  2. Instance has tags: Project=${PROJECT_NAME}, Environment=${ENVIRONMENT}"
        log_info "  3. AWS credentials are configured"
        exit 1
    fi
    
    local instance_id
    instance_id=$(echo "${instance_info}" | cut -d'|' -f1)
    local instance_ip
    instance_ip=$(echo "${instance_info}" | cut -d'|' -f2)
    
    # Output in different formats
    case "${1:-}" in
        --json)
            echo "{\"instance_id\":\"${instance_id}\",\"instance_ip\":\"${instance_ip}\"}"
            ;;
        --id)
            echo "${instance_id}"
            ;;
        --ip)
            echo "${instance_ip}"
            ;;
        *)
            echo "Instance ID: ${instance_id}"
            echo "Instance IP: ${instance_ip}"
            ;;
    esac
}

main "$@"

