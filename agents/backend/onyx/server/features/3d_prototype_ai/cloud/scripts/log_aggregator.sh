#!/bin/bash
# Log aggregation script
# Collects and aggregates logs from multiple sources

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
readonly LOG_DIR="${LOG_DIR:-./logs}"
readonly LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-30}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Aggregate and manage application logs.

COMMANDS:
    collect              Collect logs from instance
    aggregate            Aggregate logs from multiple sources
    analyze              Analyze log patterns
    search PATTERN       Search logs for pattern
    tail                 Tail live logs
    archive              Archive old logs
    clean                Clean old logs

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -d, --log-dir DIR        Log directory (default: ./logs)
    -r, --retention DAYS      Log retention days (default: 30)
    -h, --help               Show this help message

EXAMPLES:
    $0 collect --ip 1.2.3.4
    $0 search "error" --ip 1.2.3.4
    $0 tail --ip 1.2.3.4
    $0 archive --retention 90

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    SEARCH_PATTERN=""
    
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
            -d|--log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                LOG_RETENTION_DAYS="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            collect|aggregate|analyze|search|tail|archive|clean)
                COMMAND="$1"
                if [ "$COMMAND" = "search" ]; then
                    SEARCH_PATTERN="$2"
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

# Collect logs
collect_logs() {
    local ip="${1}"
    local key_path="${2}"
    local log_dir="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Collecting logs from ${ip}..."
    
    mkdir -p "${log_dir}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local log_archive="${log_dir}/logs_${ip}_${timestamp}.tar.gz"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
cd /opt/3d-prototype-ai

# Collect all logs
LOG_TEMP=\$(mktemp -d)
mkdir -p "\${LOG_TEMP}/logs"

# Application logs
if [ -d "logs" ]; then
    cp -r logs/* "\${LOG_TEMP}/logs/" 2>/dev/null || true
fi

# Docker logs
if [ -f "docker-compose.yml" ]; then
    docker-compose logs --no-color > "\${LOG_TEMP}/docker.logs" 2>/dev/null || true
fi

# System logs
sudo journalctl -u 3d-prototype-ai --no-pager > "\${LOG_TEMP}/systemd.logs" 2>/dev/null || true
sudo journalctl --since "24 hours ago" --no-pager > "\${LOG_TEMP}/system.logs" 2>/dev/null || true

# Create archive
tar -czf "\${LOG_TEMP}/logs.tar.gz" -C "\${LOG_TEMP}" .
echo "\${LOG_TEMP}/logs.tar.gz"
REMOTE_EOF
    
    # Download archive
    scp -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip}:/tmp/logs.tar.gz "${log_archive}" 2>/dev/null || log_warn "Could not download logs"
    
    log_info "Logs collected: ${log_archive}"
}

# Aggregate logs
aggregate_logs() {
    local log_dir="${1}"
    
    log_info "Aggregating logs..."
    
    mkdir -p "${log_dir}/aggregated"
    local aggregated_file="${log_dir}/aggregated/all_logs_$(date +%Y%m%d_%H%M%S).log"
    
    # Combine all log files
    find "${log_dir}" -name "*.log" -type f -exec cat {} \; > "${aggregated_file}" 2>/dev/null || true
    
    log_info "Aggregated logs: ${aggregated_file}"
}

# Analyze logs
analyze_logs() {
    local log_dir="${1}"
    
    log_info "Analyzing logs..."
    
    local analysis_file="${log_dir}/analysis_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Log Analysis Report"
        echo "==================="
        echo "Generated: $(date)"
        echo ""
        
        # Error count
        echo "Error Summary:"
        find "${log_dir}" -name "*.log" -type f -exec grep -i "error" {} \; | wc -l | xargs echo "  Total errors:"
        echo ""
        
        # Warning count
        echo "Warning Summary:"
        find "${log_dir}" -name "*.log" -type f -exec grep -i "warn" {} \; | wc -l | xargs echo "  Total warnings:"
        echo ""
        
        # Top errors
        echo "Top Error Messages:"
        find "${log_dir}" -name "*.log" -type f -exec grep -i "error" {} \; | \
            sort | uniq -c | sort -rn | head -10
        echo ""
        
        # Log file sizes
        echo "Log File Sizes:"
        find "${log_dir}" -name "*.log" -type f -exec ls -lh {} \; | \
            awk '{print $9, "(" $5 ")"}'
        
    } > "${analysis_file}"
    
    log_info "Analysis saved to: ${analysis_file}"
    cat "${analysis_file}"
}

# Search logs
search_logs() {
    local pattern="${1}"
    local ip="${2}"
    local key_path="${3}"
    
    if [ -z "${pattern}" ]; then
        error_exit 1 "Search pattern is required"
    fi
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Searching for: ${pattern}"
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
cd /opt/3d-prototype-ai

echo "Search Results for: ${pattern}"
echo "================================"
echo ""

# Search in application logs
if [ -d "logs" ]; then
    echo "Application Logs:"
    grep -r -i "${pattern}" logs/ 2>/dev/null | head -20 || echo "  No matches"
    echo ""
fi

# Search in Docker logs
if [ -f "docker-compose.yml" ]; then
    echo "Docker Logs:"
    docker-compose logs | grep -i "${pattern}" | head -20 || echo "  No matches"
    echo ""
fi

# Search in system logs
echo "System Logs:"
sudo journalctl -u 3d-prototype-ai | grep -i "${pattern}" | head -20 || echo "  No matches"
REMOTE_EOF
}

# Tail live logs
tail_logs() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Tailing live logs (Ctrl+C to stop)..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
cd /opt/3d-prototype-ai

if [ -f "docker-compose.yml" ]; then
    docker-compose logs -f
else
    sudo journalctl -u 3d-prototype-ai -f
fi
REMOTE_EOF
}

# Archive logs
archive_logs() {
    local log_dir="${1}"
    local retention_days="${2}"
    
    log_info "Archiving logs older than ${retention_days} days..."
    
    find "${log_dir}" -name "*.log" -type f -mtime +${retention_days} -exec gzip {} \; 2>/dev/null || true
    
    log_info "Logs archived"
}

# Clean old logs
clean_logs() {
    local log_dir="${1}"
    local retention_days="${2}"
    
    log_info "Cleaning logs older than ${retention_days} days..."
    
    find "${log_dir}" -name "*.log" -type f -mtime +${retention_days} -delete 2>/dev/null || true
    find "${log_dir}" -name "*.gz" -type f -mtime +${retention_days} -delete 2>/dev/null || true
    
    log_info "Old logs cleaned"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        collect)
            collect_logs "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${LOG_DIR}"
            ;;
        aggregate)
            aggregate_logs "${LOG_DIR}"
            ;;
        analyze)
            analyze_logs "${LOG_DIR}"
            ;;
        search)
            search_logs "${SEARCH_PATTERN}" "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        tail)
            tail_logs "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        archive)
            archive_logs "${LOG_DIR}" "${LOG_RETENTION_DAYS}"
            ;;
        clean)
            clean_logs "${LOG_DIR}" "${LOG_RETENTION_DAYS}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


