#!/bin/bash
# Report generator script
# Generates comprehensive deployment reports

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
readonly REPORT_DIR="${REPORT_DIR:-./reports}"
readonly REPORT_FORMAT="${REPORT_FORMAT:-html}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Generate comprehensive deployment reports.

COMMANDS:
    deployment          Generate deployment report
    performance         Generate performance report
    security            Generate security report
    cost                Generate cost report
    compliance          Generate compliance report
    comprehensive       Generate all reports

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -o, --output DIR         Output directory (default: ./reports)
    -f, --format FORMAT       Report format (html|json|txt) (default: html)
    -d, --days DAYS           Number of days for report (default: 30)
    -h, --help               Show this help message

EXAMPLES:
    $0 deployment --ip 1.2.3.4
    $0 comprehensive --format json
    $0 cost --days 90

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    DAYS=30
    
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
                REPORT_DIR="$2"
                shift 2
                ;;
            -f|--format)
                REPORT_FORMAT="$2"
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
            deployment|performance|security|cost|compliance|comprehensive)
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

# Generate deployment report
generate_deployment_report() {
    local ip="${1}"
    local key_path="${2}"
    local output_dir="${3}"
    local format="${4}"
    
    mkdir -p "${output_dir}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local report_file="${output_dir}/deployment_report_${timestamp}.${format}"
    
    log_info "Generating deployment report..."
    
    # Collect data
    local app_health
    app_health=$(curl -sf "http://${ip}:8030/health" 2>/dev/null && echo "healthy" || echo "unhealthy")
    
    local deploy_info
    deploy_info=$(aws ec2 describe-instances \
        --filters "Name=ip-address,Values=${ip}" \
        --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,LaunchTime,State.Name]' \
        --output text \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null || echo "N/A N/A N/A N/A")
    
    if [ "${format}" = "html" ]; then
        cat > "${report_file}" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Deployment Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .section { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .status { padding: 5px 10px; border-radius: 3px; display: inline-block; }
        .healthy { background: #4CAF50; color: white; }
        .unhealthy { background: #f44336; color: white; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>Deployment Report</h1>
    <p>Generated: $(date)</p>
    
    <div class="section">
        <h2>Application Status</h2>
        <p>Status: <span class="status ${app_health}">${app_health}</span></p>
    </div>
    
    <div class="section">
        <h2>Instance Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Instance ID</td><td>$(echo ${deploy_info} | awk '{print $1}')</td></tr>
            <tr><td>Instance Type</td><td>$(echo ${deploy_info} | awk '{print $2}')</td></tr>
            <tr><td>Launch Time</td><td>$(echo ${deploy_info} | awk '{print $3}')</td></tr>
            <tr><td>State</td><td>$(echo ${deploy_info} | awk '{print $4}')</td></tr>
        </table>
    </div>
</body>
</html>
EOF
    elif [ "${format}" = "json" ]; then
        cat > "${report_file}" << EOF
{
  "report_type": "deployment",
  "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "application": {
    "status": "${app_health}",
    "endpoint": "http://${ip}:8030"
  },
  "instance": {
    "id": "$(echo ${deploy_info} | awk '{print $1}')",
    "type": "$(echo ${deploy_info} | awk '{print $2}')",
    "launch_time": "$(echo ${deploy_info} | awk '{print $3}')",
    "state": "$(echo ${deploy_info} | awk '{print $4}')"
  }
}
EOF
    else
        cat > "${report_file}" << EOF
Deployment Report
=================
Generated: $(date)

Application Status: ${app_health}

Instance Information:
  Instance ID: $(echo ${deploy_info} | awk '{print $1}')
  Instance Type: $(echo ${deploy_info} | awk '{print $2}')
  Launch Time: $(echo ${deploy_info} | awk '{print $3}')
  State: $(echo ${deploy_info} | awk '{print $4}')
EOF
    fi
    
    log_info "Deployment report saved to: ${report_file}"
    echo "${report_file}"
}

# Generate performance report
generate_performance_report() {
    local ip="${1}"
    local key_path="${2}"
    local output_dir="${3}"
    local format="${4}"
    
    mkdir -p "${output_dir}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local report_file="${output_dir}/performance_report_${timestamp}.${format}"
    
    log_info "Generating performance report..."
    
    # Collect metrics
    local metrics
    metrics=$(ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << 'REMOTE_EOF' 2>/dev/null || echo "N/A|N/A|N/A|N/A"
CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.1f", 100 - $1}')
MEM=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
DISK=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
echo "${CPU}|${MEM}|${DISK}|${LOAD}"
REMOTE_EOF
)
    
    local cpu
    cpu=$(echo "${metrics}" | cut -d'|' -f1)
    local mem
    mem=$(echo "${metrics}" | cut -d'|' -f2)
    local disk
    disk=$(echo "${metrics}" | cut -d'|' -f3)
    local load
    load=$(echo "${metrics}" | cut -d'|' -f4)
    
    if [ "${format}" = "json" ]; then
        cat > "${report_file}" << EOF
{
  "report_type": "performance",
  "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "metrics": {
    "cpu": "${cpu}",
    "memory": "${mem}",
    "disk": "${disk}",
    "load": "${load}"
  }
}
EOF
    else
        cat > "${report_file}" << EOF
Performance Report
==================
Generated: $(date)

Metrics:
  CPU Usage: ${cpu}%
  Memory Usage: ${mem}%
  Disk Usage: ${disk}%
  Load Average: ${load}
EOF
    fi
    
    log_info "Performance report saved to: ${report_file}"
    echo "${report_file}"
}

# Generate comprehensive report
generate_comprehensive_report() {
    local ip="${1}"
    local key_path="${2}"
    local output_dir="${3}"
    local format="${4}"
    local days="${5}"
    
    log_info "Generating comprehensive report..."
    
    generate_deployment_report "${ip}" "${key_path}" "${output_dir}" "${format}"
    generate_performance_report "${ip}" "${key_path}" "${output_dir}" "${format}"
    
    # Generate cost report if AWS CLI available
    if command -v aws &> /dev/null; then
        log_info "Generating cost report..."
        ./scripts/cost_optimizer.sh report --days "${days}" --output "${output_dir}" || true
    fi
    
    # Generate security report
    log_info "Generating security report..."
    ./scripts/security_hardening.sh audit --ip "${ip}" --key-path "${key_path}" > "${output_dir}/security_report_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true
    
    log_info "Comprehensive report generated in: ${output_dir}"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        if [ "${COMMAND}" != "cost" ]; then
            error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required for this command"
        fi
    fi
    
    case "${COMMAND}" in
        deployment)
            generate_deployment_report "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${REPORT_DIR}" "${REPORT_FORMAT}"
            ;;
        performance)
            generate_performance_report "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${REPORT_DIR}" "${REPORT_FORMAT}"
            ;;
        security)
            ./scripts/security_hardening.sh audit --ip "${INSTANCE_IP}" --key-path "${AWS_KEY_PATH}" > "${REPORT_DIR}/security_report_$(date +%Y%m%d_%H%M%S).txt"
            ;;
        cost)
            ./scripts/cost_optimizer.sh report --days "${DAYS}" --output "${REPORT_DIR}"
            ;;
        compliance)
            ./scripts/security_hardening.sh check-compliance --ip "${INSTANCE_IP}" --key-path "${AWS_KEY_PATH}" > "${REPORT_DIR}/compliance_report_$(date +%Y%m%d_%H%M%S).txt"
            ;;
        comprehensive)
            generate_comprehensive_report "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${REPORT_DIR}" "${REPORT_FORMAT}" "${DAYS}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


