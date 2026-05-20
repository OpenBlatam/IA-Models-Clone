#!/bin/bash
# Metrics collection script
# Collects and displays deployment metrics

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
readonly METRICS_OUTPUT="${METRICS_OUTPUT:-${CLOUD_DIR}/metrics}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Collect and display deployment metrics.

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -o, --output DIR         Output directory for metrics
    -j, --json               Output in JSON format
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem
    $0 --ip 1.2.3.4 --json > metrics.json

EOF
}

# Parse arguments
parse_args() {
    JSON_OUTPUT=false
    
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
            -o|--output)
                METRICS_OUTPUT="$2"
                shift 2
                ;;
            -j|--json)
                JSON_OUTPUT=true
                shift
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

# Collect application metrics
collect_app_metrics() {
    local ip="${1}"
    
    local metrics_json
    metrics_json=$(curl -s "http://${ip}:8030/metrics" 2>/dev/null || echo "{}")
    
    echo "${metrics_json}"
}

# Collect system metrics
collect_system_metrics() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${key_path}" ]; then
        echo "{}"
        return 0
    fi
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF' 2>/dev/null || echo "{}"
# Collect system metrics
cat << SYS_METRICS
{
  "cpu": {
    "usage": $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}'),
    "load": "$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')"
  },
  "memory": {
    "total": "$(free -h | awk '/^Mem:/ {print $2}')",
    "used": "$(free -h | awk '/^Mem:/ {print $3}')",
    "free": "$(free -h | awk '/^Mem:/ {print $4}')",
    "percent": $(free | awk '/^Mem:/ {printf "%.2f", $3/$2 * 100}')
  },
  "disk": {
    "total": "$(df -h / | tail -1 | awk '{print $2}')",
    "used": "$(df -h / | tail -1 | awk '{print $3}')",
    "free": "$(df -h / | tail -1 | awk '{print $4}')",
    "percent": "$(df -h / | tail -1 | awk '{print $5}')"
  },
  "uptime": "$(uptime -p 2>/dev/null || uptime | awk -F'up' '{print $2}')"
}
SYS_METRICS
REMOTE_EOF
}

# Collect deployment metrics
collect_deployment_metrics() {
    local ip="${1}"
    
    if ! command -v aws &> /dev/null; then
        echo "{}"
        return 0
    fi
    
    local instance_id
    instance_id=$(aws ec2 describe-instances \
        --filters "Name=ip-address,Values=${ip}" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "")
    
    if [ -z "${instance_id}" ] || [ "${instance_id}" = "None" ]; then
        echo "{}"
        return 0
    fi
    
    local tags
    tags=$(aws ec2 describe-instances \
        --instance-ids "${instance_id}" \
        --query 'Reservations[0].Instances[0].Tags' \
        --output json \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "[]")
    
    local last_deploy
    last_deploy=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeployment") | .Value' 2>/dev/null || echo "")
    local last_commit
    last_commit=$(echo "${tags}" | jq -r '.[] | select(.Key=="LastDeploymentCommit") | .Value' 2>/dev/null || echo "")
    
    cat << EOF
{
  "instance_id": "${instance_id}",
  "last_deployment": "${last_deploy}",
  "last_commit": "${last_commit}"
}
EOF
}

# Display metrics
display_metrics() {
    local ip="${1}"
    local key_path="${2}"
    local json_output="${3}"
    
    if [ "${json_output}" = "true" ]; then
        # JSON output
        local app_metrics
        app_metrics=$(collect_app_metrics "${ip}")
        local system_metrics
        system_metrics=$(collect_system_metrics "${ip}" "${key_path}")
        local deploy_metrics
        deploy_metrics=$(collect_deployment_metrics "${ip}")
        
        jq -s '.[0] * .[1] * .[2]' \
            <(echo "${app_metrics}") \
            <(echo "${system_metrics}") \
            <(echo "${deploy_metrics}") 2>/dev/null || echo "{}"
    else
        # Human-readable output
        echo ""
        echo "${GREEN}==========================================${NC}"
        echo "${GREEN}Deployment Metrics${NC}"
        echo "${GREEN}==========================================${NC}"
        echo "Instance: ${ip}"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        # Application metrics
        log_info "Application Metrics:"
        local health_status
        health_status=$(curl -sf "http://${ip}:8030/health" 2>/dev/null && echo "✓ Healthy" || echo "✗ Unhealthy")
        echo "  Health: ${health_status}"
        
        local response_time
        response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://${ip}:8030/health" 2>/dev/null || echo "N/A")
        echo "  Response Time: ${response_time}s"
        echo ""
        
        # System metrics
        if [ -n "${key_path}" ]; then
            log_info "System Metrics:"
            local system_metrics
            system_metrics=$(collect_system_metrics "${ip}" "${key_path}")
            echo "${system_metrics}" | jq -r '
              "  CPU Usage: " + (.cpu.usage | tostring) + "%",
              "  Load Average: " + .cpu.load,
              "  Memory: " + .memory.used + " / " + .memory.total + " (" + (.memory.percent | tostring) + "%)",
              "  Disk: " + .disk.used + " / " + .disk.total + " (" + .disk.percent + ")",
              "  Uptime: " + .uptime
            ' 2>/dev/null || echo "  Could not retrieve system metrics"
        fi
        echo ""
        
        # Deployment metrics
        log_info "Deployment Metrics:"
        local deploy_metrics
        deploy_metrics=$(collect_deployment_metrics "${ip}")
        echo "${deploy_metrics}" | jq -r '
          "  Instance ID: " + .instance_id,
          "  Last Deployment: " + .last_deployment,
          "  Last Commit: " + .last_commit
        ' 2>/dev/null || echo "  Could not retrieve deployment metrics"
        
        echo ""
        echo "${GREEN}==========================================${NC}"
    fi
}

# Save metrics
save_metrics() {
    local metrics="${1}"
    local output_dir="${2}"
    
    mkdir -p "${output_dir}"
    
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local metrics_file="${output_dir}/metrics_${timestamp}.json"
    
    echo "${metrics}" > "${metrics_file}"
    log_info "Metrics saved to: ${metrics_file}"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "INSTANCE_IP is required"
    fi
    
    validate_ip "${INSTANCE_IP}"
    
    if [ "${JSON_OUTPUT}" = "true" ]; then
        local app_metrics
        app_metrics=$(collect_app_metrics "${INSTANCE_IP}")
        local system_metrics
        system_metrics=$(collect_system_metrics "${INSTANCE_IP}" "${AWS_KEY_PATH}")
        local deploy_metrics
        deploy_metrics=$(collect_deployment_metrics "${INSTANCE_IP}")
        
        jq -s '.[0] * .[1] * .[2]' \
            <(echo "${app_metrics}") \
            <(echo "${system_metrics}") \
            <(echo "${deploy_metrics}") 2>/dev/null || echo "{}"
    else
        display_metrics "${INSTANCE_IP}" "${AWS_KEY_PATH}" "false"
        
        # Save metrics
        local app_metrics
        app_metrics=$(collect_app_metrics "${INSTANCE_IP}")
        local system_metrics
        system_metrics=$(collect_system_metrics "${INSTANCE_IP}" "${AWS_KEY_PATH}")
        local deploy_metrics
        deploy_metrics=$(collect_deployment_metrics "${INSTANCE_IP}")
        
        local all_metrics
        all_metrics=$(jq -s '.[0] * .[1] * .[2]' \
            <(echo "${app_metrics}") \
            <(echo "${system_metrics}") \
            <(echo "${deploy_metrics}") 2>/dev/null || echo "{}")
        
        save_metrics "${all_metrics}" "${METRICS_OUTPUT}"
    fi
}

main "$@"


