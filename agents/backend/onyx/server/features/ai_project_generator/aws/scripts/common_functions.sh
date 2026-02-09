#!/bin/bash

###############################################################################
# Common Functions Library for Automation Scripts
# Shared utilities and helper functions
###############################################################################

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

###############################################################################
# Logging Functions
###############################################################################

log_info() {
    local message="$1"
    local log_file="${LOG_FILE:-/var/log/automation.log}"
    echo -e "${GREEN}[INFO]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] ${message}" | \
        tee -a "${log_file}"
}

log_warn() {
    local message="$1"
    local log_file="${LOG_FILE:-/var/log/automation.log}"
    echo -e "${YELLOW}[WARN]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] ${message}" | \
        tee -a "${log_file}"
}

log_error() {
    local message="$1"
    local log_file="${LOG_FILE:-/var/log/automation.log}"
    echo -e "${RED}[ERROR]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] ${message}" | \
        tee -a "${log_file}" >&2
}

log_debug() {
    local message="$1"
    local log_file="${LOG_FILE:-/var/log/automation.log}"
    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] ${message}" | \
            tee -a "${log_file}"
    fi
}

log_success() {
    local message="$1"
    local log_file="${LOG_FILE:-/var/log/automation.log}"
    echo -e "${GREEN}[SUCCESS]${NC} [$(date +'%Y-%m-%d %H:%M:%S')] ${message}" | \
        tee -a "${log_file}"
}

###############################################################################
# Validation Functions
###############################################################################

check_command() {
    local cmd="$1"
    if ! command -v "${cmd}" &> /dev/null; then
        log_error "Command not found: ${cmd}"
        return 1
    fi
    return 0
}

check_file_exists() {
    local file_path="$1"
    if [ ! -f "${file_path}" ]; then
        log_error "File not found: ${file_path}"
        return 1
    fi
    return 0
}

check_directory_exists() {
    local dir_path="$1"
    if [ ! -d "${dir_path}" ]; then
        log_error "Directory not found: ${dir_path}"
        return 1
    fi
    return 0
}

check_disk_space() {
    local path="$1"
    local required_gb="${2:-1}"
    
    local available_gb
    available_gb=$(df -BG "${path}" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "${available_gb}" -lt "${required_gb}" ]; then
        log_error "Insufficient disk space: ${available_gb}GB available, ${required_gb}GB required"
        return 1
    fi
    
    log_debug "Disk space check passed: ${available_gb}GB available"
    return 0
}

###############################################################################
# AWS Functions
###############################################################################

check_aws_credentials() {
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        return 1
    fi
    return 0
}

send_sns_alert() {
    local topic_arn="$1"
    local subject="$2"
    local message="$3"
    
    if [ -z "${topic_arn}" ]; then
        return 0
    fi
    
    if ! check_aws_credentials; then
        log_warn "Cannot send SNS alert: AWS credentials not configured"
        return 1
    fi
    
    aws sns publish \
        --topic-arn "${topic_arn}" \
        --subject "${subject}" \
        --message "${message}" \
        > /dev/null 2>&1 || {
        log_warn "Failed to send SNS alert"
        return 1
    }
    
    log_debug "SNS alert sent successfully"
    return 0
}

send_cloudwatch_metric() {
    local namespace="$1"
    local metric_name="$2"
    local value="$3"
    local unit="${4:-Count}"
    local dimensions="${5:-}"
    
    if ! check_aws_credentials; then
        log_debug "Cannot send CloudWatch metric: AWS credentials not configured"
        return 0
    fi
    
    local cmd="aws cloudwatch put-metric-data \
        --namespace '${namespace}' \
        --metric-name '${metric_name}' \
        --value ${value} \
        --unit '${unit}'"
    
    if [ -n "${dimensions}" ]; then
        cmd="${cmd} --dimensions '${dimensions}'"
    fi
    
    eval "${cmd}" > /dev/null 2>&1 || {
        log_debug "Failed to send CloudWatch metric"
        return 1
    }
    
    log_debug "CloudWatch metric sent: ${namespace}/${metric_name}=${value}"
    return 0
}

upload_to_s3() {
    local local_path="$1"
    local s3_path="$2"
    local bucket="$3"
    
    if [ -z "${bucket}" ]; then
        log_warn "S3 bucket not specified"
        return 1
    fi
    
    if ! check_aws_credentials; then
        log_warn "Cannot upload to S3: AWS credentials not configured"
        return 1
    fi
    
    aws s3 cp "${local_path}" "${s3_path}" --quiet || {
        log_error "Failed to upload to S3: ${local_path}"
        return 1
    }
    
    log_debug "Uploaded to S3: ${s3_path}"
    return 0
}

###############################################################################
# System Functions
###############################################################################

get_cpu_usage() {
    if command -v top &> /dev/null; then
        top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | \
            awk '{print 100 - $1}'
    elif [ -f /proc/loadavg ]; then
        # Fallback method
        awk '{print $1 * 100}' /proc/loadavg
    else
        echo "0"
    fi
}

get_memory_usage() {
    free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}'
}

get_disk_usage() {
    local path="${1:-/}"
    df -h "${path}" | awk 'NR==2 {print $5}' | sed 's/%//'
}

get_disk_available() {
    local path="${1:-/}"
    df -BG "${path}" | awk 'NR==2 {print $4}' | sed 's/G//'
}

###############################################################################
# Network Functions
###############################################################################

check_url() {
    local url="$1"
    local timeout="${2:-10}"
    local max_retries="${3:-3}"
    
    local retries=0
    while [ $retries -lt $max_retries ]; do
        if curl -f -s --max-time "${timeout}" "${url}" > /dev/null 2>&1; then
            return 0
        fi
        retries=$((retries + 1))
        sleep 1
    done
    
    return 1
}

get_response_time() {
    local url="$1"
    local timeout="${2:-10}"
    
    curl -s -o /dev/null -w "%{time_total}" \
        --max-time "${timeout}" "${url}" 2>/dev/null || echo "0"
}

get_http_status() {
    local url="$1"
    local timeout="${2:-10}"
    
    curl -s -o /dev/null -w "%{http_code}" \
        --max-time "${timeout}" "${url}" 2>/dev/null || echo "000"
}

###############################################################################
# File Functions
###############################################################################

create_backup_marker() {
    local backup_path="$1"
    local marker_file="${backup_path}/.backup_marker"
    
    cat > "${marker_file}" <<EOF
BACKUP_DATE=$(date -Iseconds)
BACKUP_HOSTNAME=$(hostname)
BACKUP_USER=$(whoami)
BACKUP_VERSION=1.0
EOF
    
    echo "${marker_file}"
}

verify_backup_integrity() {
    local backup_path="$1"
    local marker_file="${backup_path}/.backup_marker"
    
    if [ ! -f "${marker_file}" ]; then
        log_warn "Backup marker not found: ${marker_file}"
        return 1
    fi
    
    # Check if backup directory is not empty
    if [ -z "$(ls -A "${backup_path}" 2>/dev/null)" ]; then
        log_error "Backup directory is empty"
        return 1
    fi
    
    log_debug "Backup integrity verified"
    return 0
}

calculate_directory_size() {
    local dir_path="$1"
    du -sh "${dir_path}" 2>/dev/null | cut -f1
}

###############################################################################
# Process Functions
###############################################################################

is_process_running() {
    local process_name="$1"
    pgrep -f "${process_name}" > /dev/null 2>&1
}

get_process_count() {
    local process_name="$1"
    pgrep -f "${process_name}" | wc -l
}

###############################################################################
# Docker Functions
###############################################################################

check_docker_container() {
    local container_name="$1"
    docker ps --format "{{.Names}}" | grep -q "^${container_name}$"
}

get_docker_container_status() {
    local container_name="$1"
    docker inspect --format='{{.State.Status}}' "${container_name}" 2>/dev/null || echo "not_found"
}

get_docker_container_health() {
    local container_name="$1"
    docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "no_healthcheck"
}

###############################################################################
# Utility Functions
###############################################################################

generate_report() {
    local report_file="$1"
    local title="$2"
    shift 2
    local content="$@"
    
    cat > "${report_file}" <<EOF
========================================
${title}
========================================
Generated: $(date -Iseconds)
Hostname: $(hostname)
User: $(whoami)

${content}

========================================
EOF
    
    log_debug "Report generated: ${report_file}"
    echo "${report_file}"
}

format_duration() {
    local seconds="$1"
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

retry_command() {
    local max_attempts="$1"
    local delay="$2"
    shift 2
    local cmd="$@"
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if eval "${cmd}"; then
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            log_debug "Command failed, retrying in ${delay}s (attempt ${attempt}/${max_attempts})"
            sleep "${delay}"
        fi
        
        attempt=$((attempt + 1))
    done
    
    log_error "Command failed after ${max_attempts} attempts: ${cmd}"
    return 1
}

###############################################################################
# Export Functions
###############################################################################

# Export all functions for use in other scripts
export -f log_info log_warn log_error log_debug log_success
export -f check_command check_file_exists check_directory_exists check_disk_space
export -f check_aws_credentials send_sns_alert send_cloudwatch_metric upload_to_s3
export -f get_cpu_usage get_memory_usage get_disk_usage get_disk_available
export -f check_url get_response_time get_http_status
export -f create_backup_marker verify_backup_integrity calculate_directory_size
export -f is_process_running get_process_count
export -f check_docker_container get_docker_container_status get_docker_container_health
export -f generate_report format_duration retry_command

