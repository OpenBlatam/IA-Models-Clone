#!/bin/bash
# Deployment status script
# Shows current deployment status and history

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Show deployment status and history.

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem
    $0  # Uses INSTANCE_IP from .env

EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            -k|--key-path)
                AWS_KEY_PATH="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Get deployment info from EC2 tags
get_deployment_info() {
    local instance_id="${1}"
    
    if ! command -v aws &> /dev/null; then
        log_warn "AWS CLI not available, cannot get deployment info"
        return 1
    fi
    
    local tags
    tags=$(aws ec2 describe-instances \
        --instance-ids "${instance_id}" \
        --query 'Reservations[0].Instances[0].Tags' \
        --output json \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "[]")
    
    local last_deployment
    last_deployment=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeployment") | .Value' 2>/dev/null || echo "unknown")
    local last_commit
    last_commit=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeploymentCommit") | .Value' 2>/dev/null || echo "unknown")
    local last_by
    last_by=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeploymentBy") | .Value' 2>/dev/null || echo "unknown")
    local last_branch
    last_branch=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeploymentBranch") | .Value' 2>/dev/null || echo "unknown")
    
    echo "${last_deployment}|${last_commit}|${last_by}|${last_branch}"
}

# Get instance info
get_instance_info() {
    if [ -z "${INSTANCE_IP}" ]; then
        log_error "INSTANCE_IP is required"
        return 1
    fi
    
    # Try to get instance ID from AWS
    local instance_id
    instance_id=$(aws ec2 describe-instances \
        --filters "Name=ip-address,Values=${INSTANCE_IP}" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "")
    
    echo "${instance_id}"
}

# Get application status
get_application_status() {
    local ip="${1}"
    
    local health_status
    health_status=$(curl -sf -m 5 "http://${ip}:8030/health" 2>/dev/null && echo "healthy" || echo "unhealthy")
    
    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://${ip}:8030/health" 2>/dev/null || echo "N/A")
    
    echo "${health_status}|${response_time}"
}

# Get system status via SSH
get_system_status() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${key_path}" ]; then
        echo "N/A|N/A|N/A|N/A"
        return 0
    fi
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 \
        ubuntu@${ip} << 'REMOTE_EOF' 2>/dev/null || echo "N/A|N/A|N/A|N/A"
# Get system info
UPTIME=$(uptime -p 2>/dev/null || uptime | awk -F'up' '{print $2}' | awk '{print $1,$2}')
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
MEMORY=$(free -h | awk '/^Mem:/ {print $3"/"$2" ("$5" used)"}')
DISK=$(df -h / | tail -1 | awk '{print $5" used ("$4" free)"}')

echo "${UPTIME}|${LOAD}|${MEMORY}|${DISK}"
REMOTE_EOF
}

# Display status
display_status() {
    local ip="${1}"
    local key_path="${2}"
    
    clear
    cat << EOF

${GREEN}==========================================${NC}
${GREEN}Deployment Status${NC}
${GREEN}==========================================${NC}
Instance IP: ${ip}
Time: $(date '+%Y-%m-%d %H:%M:%S')
${GREEN}==========================================${NC}

EOF
    
    # Get application status
    log_info "Checking application status..."
    local app_status
    app_status=$(get_application_status "${ip}")
    local health
    health=$(echo "${app_status}" | cut -d'|' -f1)
    local response_time
    response_time=$(echo "${app_status}" | cut -d'|' -f2)
    
    if [ "${health}" = "healthy" ]; then
        log_info "Application: ${GREEN}✓ Healthy${NC} (Response time: ${response_time}s)"
    else
        log_error "Application: ✗ Unhealthy"
    fi
    
    echo ""
    
    # Get system status
    log_info "Checking system status..."
    local system_status
    system_status=$(get_system_status "${ip}" "${key_path}")
    local uptime
    uptime=$(echo "${system_status}" | cut -d'|' -f1)
    local load
    load=$(echo "${system_status}" | cut -d'|' -f2)
    local memory
    memory=$(echo "${system_status}" | cut -d'|' -f3)
    local disk
    disk=$(echo "${system_status}" | cut -d'|' -f4)
    
    if [ "${uptime}" != "N/A" ]; then
        printf "Uptime:     %s\n" "${uptime}"
        printf "Load:       %s\n" "${load}"
        printf "Memory:     %s\n" "${memory}"
        printf "Disk:       %s\n" "${disk}"
    else
        log_warn "Could not get system status (SSH may not be configured)"
    fi
    
    echo ""
    
    # Get deployment info
    if command -v aws &> /dev/null && check_aws_credentials; then
        log_info "Getting deployment history..."
        local instance_id
        instance_id=$(get_instance_info)
        
        if [ -n "${instance_id}" ] && [ "${instance_id}" != "None" ]; then
            local deploy_info
            deploy_info=$(get_deployment_info "${instance_id}")
            local last_deploy
            last_deploy=$(echo "${deploy_info}" | cut -d'|' -f1)
            local last_commit
            last_commit=$(echo "${deploy_info}" | cut -d'|' -f2)
            local last_by
            last_by=$(echo "${deploy_info}" | cut -d'|' -f3)
            local last_branch
            last_branch=$(echo "${deploy_info}" | cut -d'|' -f4)
            
            if [ "${last_deploy}" != "unknown" ]; then
                echo "${GREEN}Last Deployment:${NC}"
                printf "  Time:    %s\n" "${last_deploy}"
                printf "  Commit:  %s\n" "${last_commit}"
                printf "  By:      %s\n" "${last_by}"
                printf "  Branch:  %s\n" "${last_branch}"
            fi
        fi
    fi
    
    echo ""
    echo "${GREEN}==========================================${NC}"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "INSTANCE_IP is required. Use --ip option or set in .env"
    fi
    
    validate_ip "${INSTANCE_IP}"
    
    display_status "${INSTANCE_IP}" "${AWS_KEY_PATH}"
}

main "$@"


