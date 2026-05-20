#!/bin/bash
# Dashboard script
# Displays comprehensive deployment dashboard

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

Display comprehensive deployment dashboard.

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -r, --refresh SECONDS    Auto-refresh interval (default: 30)
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4
    $0 --ip 1.2.3.4 --refresh 60

EOF
}

# Parse arguments
parse_args() {
    REFRESH_INTERVAL=30
    
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
            -r|--refresh)
                REFRESH_INTERVAL="$2"
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

# Get application health
get_app_health() {
    local ip="${1}"
    
    local health_status
    health_status=$(curl -sf -m 5 "http://${ip}:8030/health" 2>/dev/null && echo "healthy" || echo "unhealthy")
    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://${ip}:8030/health" 2>/dev/null || echo "N/A")
    
    echo "${health_status}|${response_time}"
}

# Get system metrics
get_system_metrics() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${key_path}" ]; then
        echo "N/A|N/A|N/A|N/A|N/A"
        return 0
    fi
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 \
        ubuntu@${ip} << 'REMOTE_EOF' 2>/dev/null || echo "N/A|N/A|N/A|N/A|N/A"
CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.1f", 100 - $1}')
MEM=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
DISK=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
UPTIME=$(uptime -p 2>/dev/null || uptime | awk -F'up' '{print $2}' | awk '{print $1,$2}')
echo "${CPU}|${MEM}|${DISK}|${LOAD}|${UPTIME}"
REMOTE_EOF
}

# Get deployment info
get_deployment_info() {
    local ip="${1}"
    
    if ! command -v aws &> /dev/null; then
        echo "N/A|N/A|N/A|N/A"
        return 0
    fi
    
    local instance_id
    instance_id=$(aws ec2 describe-instances \
        --filters "Name=ip-address,Values=${ip}" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "")
    
    if [ -z "${instance_id}" ] || [ "${instance_id}" = "None" ]; then
        echo "N/A|N/A|N/A|N/A"
        return 0
    fi
    
    local tags
    tags=$(aws ec2 describe-instances \
        --instance-ids "${instance_id}" \
        --query 'Reservations[0].Instances[0].Tags' \
        --output json \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "[]")
    
    local last_deploy
    last_deploy=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeployment") | .Value' 2>/dev/null || echo "N/A")
    local last_commit
    last_commit=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeploymentCommit") | .Value' 2>/dev/null || echo "N/A")
    local last_by
    last_by=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeploymentBy") | .Value' 2>/dev/null || echo "N/A")
    local instance_type
    instance_type=$(aws ec2 describe-instances \
        --instance-ids "${instance_id}" \
        --query 'Reservations[0].Instances[0].InstanceType' \
        --output text \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "N/A")
    
    echo "${last_deploy}|${last_commit}|${last_by}|${instance_type}"
}

# Display dashboard
display_dashboard() {
    local ip="${1}"
    local key_path="${2}"
    
    clear
    cat << EOF

${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}
${GREEN}║        3D Prototype AI - Deployment Dashboard             ║${NC}
${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}

${BLUE}Instance:${NC} ${ip}
${BLUE}Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')
${BLUE}Refresh:${NC} Every ${REFRESH_INTERVAL}s (Ctrl+C to exit)

${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${GREEN}Application Status${NC}
${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

EOF
    
    # Application health
    local app_health
    app_health=$(get_app_health "${ip}")
    local health_status
    health_status=$(echo "${app_health}" | cut -d'|' -f1)
    local response_time
    response_time=$(echo "${app_health}" | cut -d'|' -f2)
    
    if [ "${health_status}" = "healthy" ]; then
        printf "  ${GREEN}✓${NC} Status: ${GREEN}Healthy${NC}\n"
        printf "  ${GREEN}✓${NC} Response Time: ${response_time}s\n"
    else
        printf "  ${RED}✗${NC} Status: ${RED}Unhealthy${NC}\n"
        printf "  ${RED}✗${NC} Response Time: ${response_time}\n"
    fi
    
    echo ""
    echo "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "${GREEN}System Resources${NC}"
    echo "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # System metrics
    local system_metrics
    system_metrics=$(get_system_metrics "${ip}" "${key_path}")
    local cpu
    cpu=$(echo "${system_metrics}" | cut -d'|' -f1)
    local mem
    mem=$(echo "${system_metrics}" | cut -d'|' -f2)
    local disk
    disk=$(echo "${system_metrics}" | cut -d'|' -f3)
    local load
    load=$(echo "${system_metrics}" | cut -d'|' -f4)
    local uptime
    uptime=$(echo "${system_metrics}" | cut -d'|' -f5)
    
    if [ "${cpu}" != "N/A" ]; then
        # CPU bar
        local cpu_int
        cpu_int=$(echo "${cpu}" | awk '{print int($1)}')
        printf "  CPU:   ["
        local i=0
        while [ $i -lt 50 ]; do
            if [ $i -lt $((cpu_int / 2)) ]; then
                if [ $cpu_int -gt 80 ]; then
                    printf "${RED}█${NC}"
                elif [ $cpu_int -gt 60 ]; then
                    printf "${YELLOW}█${NC}"
                else
                    printf "${GREEN}█${NC}"
                fi
            else
                printf " "
            fi
            i=$((i + 1))
        done
        printf "] ${cpu}%%\n"
        
        # Memory bar
        local mem_int
        mem_int=$(echo "${mem}" | awk '{print int($1)}')
        printf "  Memory:["
        i=0
        while [ $i -lt 50 ]; do
            if [ $i -lt $((mem_int / 2)) ]; then
                if [ $mem_int -gt 85 ]; then
                    printf "${RED}█${NC}"
                elif [ $mem_int -gt 70 ]; then
                    printf "${YELLOW}█${NC}"
                else
                    printf "${GREEN}█${NC}"
                fi
            else
                printf " "
            fi
            i=$((i + 1))
        done
        printf "] ${mem}%%\n"
        
        # Disk bar
        local disk_int
        disk_int=$(echo "${disk}" | awk '{print int($1)}')
        printf "  Disk:  ["
        i=0
        while [ $i -lt 50 ]; do
            if [ $i -lt $((disk_int / 2)) ]; then
                if [ $disk_int -gt 90 ]; then
                    printf "${RED}█${NC}"
                elif [ $disk_int -gt 75 ]; then
                    printf "${YELLOW}█${NC}"
                else
                    printf "${GREEN}█${NC}"
                fi
            else
                printf " "
            fi
            i=$((i + 1))
        done
        printf "] ${disk}%%\n"
        
        printf "  Load:  ${load}\n"
        printf "  Uptime: ${uptime}\n"
    else
        echo "  System metrics unavailable"
    fi
    
    echo ""
    echo "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "${GREEN}Deployment Information${NC}"
    echo "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Deployment info
    local deploy_info
    deploy_info=$(get_deployment_info "${ip}")
    local last_deploy
    last_deploy=$(echo "${deploy_info}" | cut -d'|' -f1)
    local last_commit
    last_commit=$(echo "${deploy_info}" | cut -d'|' -f2)
    local last_by
    last_by=$(echo "${deploy_info}" | cut -d'|' -f3)
    local instance_type
    instance_type=$(echo "${deploy_info}" | cut -d'|' -f4)
    
    if [ "${last_deploy}" != "N/A" ]; then
        printf "  Last Deployment: ${last_deploy}\n"
        printf "  Commit: ${last_commit}\n"
        printf "  Deployed By: ${last_by}\n"
        printf "  Instance Type: ${instance_type}\n"
    else
        echo "  Deployment info unavailable"
    fi
    
    echo ""
    echo "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "INSTANCE_IP is required"
    fi
    
    validate_ip "${INSTANCE_IP}"
    
    # Continuous refresh
    while true; do
        display_dashboard "${INSTANCE_IP}" "${AWS_KEY_PATH}"
        sleep "${REFRESH_INTERVAL}"
    done
}

main "$@"


