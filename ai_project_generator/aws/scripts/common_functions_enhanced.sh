#!/bin/bash

###############################################################################
# Enhanced Common Functions Library for Automation Scripts
# Improved version with better error handling, validation, and POSIX compliance
###############################################################################

# Ensure POSIX compliance
set -o posix 2>/dev/null || true

# Colors for output (with fallback for non-color terminals)
if [ -t 1 ]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    readonly CYAN='\033[0;36m'
    readonly MAGENTA='\033[0;35m'
    readonly NC='\033[0m' # No Color
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly BLUE=''
    readonly CYAN=''
    readonly MAGENTA=''
    readonly NC=''
fi

###############################################################################
# Configuration and Constants
###############################################################################

readonly SCRIPT_VERSION="2.1.0"
readonly DEFAULT_LOG_FILE="/var/log/automation.log"
readonly DEFAULT_TIMEOUT=30
readonly DEFAULT_RETRIES=3
readonly DEFAULT_RETRY_DELAY=5

###############################################################################
# Logging Functions (Enhanced)
###############################################################################

# Initialize logging
init_logging() {
    local log_file="${1:-${DEFAULT_LOG_FILE}}"
    local log_dir
    log_dir=$(dirname "${log_file}")
    
    # Create log directory if it doesn't exist
    if [ ! -d "${log_dir}" ]; then
        mkdir -p "${log_dir}" 2>/dev/null || {
            echo "Warning: Cannot create log directory: ${log_dir}" >&2
            log_file="/tmp/automation.log"
        }
    fi
    
    export LOG_FILE="${log_file}"
    touch "${LOG_FILE}" 2>/dev/null || {
        echo "Warning: Cannot write to log file: ${log_file}" >&2
        export LOG_FILE="/tmp/automation.log"
    }
}

log_info() {
    local message="$1"
    local log_file="${LOG_FILE:-${DEFAULT_LOG_FILE}}"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date)
    printf "%s[INFO]%s [%s] %s\n" "${GREEN}" "${NC}" "${timestamp}" "${message}" | \
        tee -a "${log_file}" 2>/dev/null || \
        printf "%s[INFO]%s [%s] %s\n" "${GREEN}" "${NC}" "${timestamp}" "${message}"
}

log_warn() {
    local message="$1"
    local log_file="${LOG_FILE:-${DEFAULT_LOG_FILE}}"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date)
    printf "%s[WARN]%s [%s] %s\n" "${YELLOW}" "${NC}" "${timestamp}" "${message}" | \
        tee -a "${log_file}" 2>/dev/null >&2 || \
        printf "%s[WARN]%s [%s] %s\n" "${YELLOW}" "${NC}" "${timestamp}" "${message}" >&2
}

log_error() {
    local message="$1"
    local log_file="${LOG_FILE:-${DEFAULT_LOG_FILE}}"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date)
    printf "%s[ERROR]%s [%s] %s\n" "${RED}" "${NC}" "${timestamp}" "${message}" | \
        tee -a "${log_file}" 2>/dev/null >&2 || \
        printf "%s[ERROR]%s [%s] %s\n" "${RED}" "${NC}" "${timestamp}" "${message}" >&2
}

log_debug() {
    local message="$1"
    if [ "${DEBUG:-false}" = "true" ] || [ "${VERBOSE:-false}" = "true" ]; then
        local log_file="${LOG_FILE:-${DEFAULT_LOG_FILE}}"
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date)
        printf "%s[DEBUG]%s [%s] %s\n" "${BLUE}" "${NC}" "${timestamp}" "${message}" | \
            tee -a "${log_file}" 2>/dev/null || \
            printf "%s[DEBUG]%s [%s] %s\n" "${BLUE}" "${NC}" "${timestamp}" "${message}"
    fi
}

log_success() {
    local message="$1"
    local log_file="${LOG_FILE:-${DEFAULT_LOG_FILE}}"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date)
    printf "%s[SUCCESS]%s [%s] %s\n" "${GREEN}" "${NC}" "${timestamp}" "${message}" | \
        tee -a "${log_file}" 2>/dev/null || \
        printf "%s[SUCCESS]%s [%s] %s\n" "${GREEN}" "${NC}" "${timestamp}" "${message}"
}

log_section() {
    local title="$1"
    local log_file="${LOG_FILE:-${DEFAULT_LOG_FILE}}"
    local separator="=========================================="
    printf "\n%s\n%s\n%s\n" "${separator}" "${title}" "${separator}" | \
        tee -a "${log_file}" 2>/dev/null || \
        printf "\n%s\n%s\n%s\n" "${separator}" "${title}" "${separator}"
}

###############################################################################
# Validation Functions (Enhanced)
###############################################################################

# Validate command exists
check_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        log_error "Command not found: ${cmd}"
        return 1
    fi
    log_debug "Command found: ${cmd}"
    return 0
}

# Validate file exists and is readable
check_file_exists() {
    local file_path="$1"
    if [ ! -f "${file_path}" ]; then
        log_error "File not found: ${file_path}"
        return 1
    fi
    if [ ! -r "${file_path}" ]; then
        log_error "File not readable: ${file_path}"
        return 1
    fi
    log_debug "File validated: ${file_path}"
    return 0
}

# Validate directory exists and is accessible
check_directory_exists() {
    local dir_path="$1"
    if [ ! -d "${dir_path}" ]; then
        log_error "Directory not found: ${dir_path}"
        return 1
    fi
    if [ ! -r "${dir_path}" ] || [ ! -x "${dir_path}" ]; then
        log_error "Directory not accessible: ${dir_path}"
        return 1
    fi
    log_debug "Directory validated: ${dir_path}"
    return 0
}

# Validate disk space with better error messages
check_disk_space() {
    local path="$1"
    local required_gb="${2:-1}"
    
    if ! check_directory_exists "$(dirname "${path}")"; then
        return 1
    fi
    
    local available_gb
    if command -v df >/dev/null 2>&1; then
        # Try different df options for portability
        available_gb=$(df -BG "${path}" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || \
            df -g "${path}" 2>/dev/null | awk 'NR==2 {print $4}' || \
            df -k "${path}" 2>/dev/null | awk 'NR==2 {print $4/1024/1024}' | cut -d. -f1)
    else
        log_error "df command not available"
        return 1
    fi
    
    # Validate numeric value
    if ! echo "${available_gb}" | grep -qE '^[0-9]+$'; then
        log_error "Cannot determine available disk space"
        return 1
    fi
    
    if [ "${available_gb}" -lt "${required_gb}" ]; then
        log_error "Insufficient disk space: ${available_gb}GB available, ${required_gb}GB required"
        return 1
    fi
    
    log_debug "Disk space check passed: ${available_gb}GB available"
    return 0
}

# Validate URL format
validate_url() {
    local url="$1"
    if echo "${url}" | grep -qE '^https?://[^[:space:]]+$'; then
        return 0
    fi
    log_error "Invalid URL format: ${url}"
    return 1
}

# Validate email format
validate_email() {
    local email="$1"
    if echo "${email}" | grep -qE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'; then
        return 0
    fi
    log_error "Invalid email format: ${email}"
    return 1
}

# Validate port number
validate_port() {
    local port="$1"
    if echo "${port}" | grep -qE '^[0-9]+$' && [ "${port}" -ge 1 ] && [ "${port}" -le 65535 ]; then
        return 0
    fi
    log_error "Invalid port number: ${port} (must be 1-65535)"
    return 1
}

# Validate numeric value
validate_number() {
    local value="$1"
    local min="${2:-0}"
    local max="${3:-999999}"
    
    if ! echo "${value}" | grep -qE '^[0-9]+$'; then
        log_error "Invalid number: ${value}"
        return 1
    fi
    
    if [ "${value}" -lt "${min}" ] || [ "${value}" -gt "${max}" ]; then
        log_error "Number out of range: ${value} (must be ${min}-${max})"
        return 1
    fi
    
    return 0
}

###############################################################################
# AWS Functions (Enhanced with better error handling)
###############################################################################

check_aws_credentials() {
    if ! check_command "aws"; then
        return 1
    fi
    
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS credentials not configured or invalid"
        return 1
    fi
    
    log_debug "AWS credentials validated"
    return 0
}

send_sns_alert() {
    local topic_arn="${1:-}"
    local subject="${2:-}"
    local message="${3:-}"
    
    if [ -z "${topic_arn}" ]; then
        log_debug "SNS topic ARN not provided, skipping alert"
        return 0
    fi
    
    if ! check_aws_credentials; then
        log_warn "Cannot send SNS alert: AWS credentials not configured"
        return 1
    fi
    
    if ! validate_url "https://sns.${AWS_REGION:-us-east-1}.amazonaws.com" 2>/dev/null; then
        log_warn "Invalid AWS region for SNS"
        return 1
    fi
    
    if aws sns publish \
        --topic-arn "${topic_arn}" \
        --subject "${subject}" \
        --message "${message}" \
        --region "${AWS_REGION:-us-east-1}" \
        >/dev/null 2>&1; then
        log_debug "SNS alert sent successfully"
        return 0
    else
        log_warn "Failed to send SNS alert"
        return 1
    fi
}

send_cloudwatch_metric() {
    local namespace="${1:-}"
    local metric_name="${2:-}"
    local value="${3:-0}"
    local unit="${4:-Count}"
    local dimensions="${5:-}"
    
    if [ -z "${namespace}" ] || [ -z "${metric_name}" ]; then
        log_debug "CloudWatch metric parameters incomplete"
        return 0
    fi
    
    if ! check_aws_credentials; then
        log_debug "Cannot send CloudWatch metric: AWS credentials not configured"
        return 0
    fi
    
    # Validate numeric value
    if ! validate_number "${value}" 0 999999999; then
        log_warn "Invalid metric value: ${value}"
        return 1
    fi
    
    local cmd="aws cloudwatch put-metric-data \
        --namespace '${namespace}' \
        --metric-name '${metric_name}' \
        --value ${value} \
        --unit '${unit}' \
        --region '${AWS_REGION:-us-east-1}'"
    
    if [ -n "${dimensions}" ]; then
        cmd="${cmd} --dimensions '${dimensions}'"
    fi
    
    if eval "${cmd}" >/dev/null 2>&1; then
        log_debug "CloudWatch metric sent: ${namespace}/${metric_name}=${value}"
        return 0
    else
        log_debug "Failed to send CloudWatch metric"
        return 1
    fi
}

upload_to_s3() {
    local local_path="$1"
    local s3_path="$2"
    local bucket="${3:-}"
    
    if [ -z "${bucket}" ]; then
        log_warn "S3 bucket not specified"
        return 1
    fi
    
    if ! check_file_exists "${local_path}"; then
        return 1
    fi
    
    if ! check_aws_credentials; then
        log_warn "Cannot upload to S3: AWS credentials not configured"
        return 1
    fi
    
    if aws s3 cp "${local_path}" "${s3_path}" \
        --region "${AWS_REGION:-us-east-1}" \
        --quiet >/dev/null 2>&1; then
        log_debug "Uploaded to S3: ${s3_path}"
        return 0
    else
        log_error "Failed to upload to S3: ${local_path}"
        return 1
    fi
}

###############################################################################
# System Functions (Enhanced with better portability)
###############################################################################

get_cpu_usage() {
    if command -v top >/dev/null 2>&1; then
        top -bn1 2>/dev/null | grep "Cpu(s)" | \
            sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | \
            awk '{print 100 - $1}' || echo "0"
    elif [ -f /proc/loadavg ]; then
        # Fallback: use load average as approximation
        awk '{print $1 * 100}' /proc/loadavg 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

get_memory_usage() {
    if command -v free >/dev/null 2>&1; then
        free 2>/dev/null | grep Mem | \
            awk '{printf "%.0f", $3/$2 * 100.0}' || echo "0"
    else
        echo "0"
    fi
}

get_disk_usage() {
    local path="${1:-/}"
    if command -v df >/dev/null 2>&1; then
        df -h "${path}" 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//' || echo "0"
    else
        echo "0"
    fi
}

get_disk_available() {
    local path="${1:-/}"
    if command -v df >/dev/null 2>&1; then
        # Try different df options for portability
        df -BG "${path}" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || \
            df -g "${path}" 2>/dev/null | awk 'NR==2 {print $4}' || \
            df -k "${path}" 2>/dev/null | awk 'NR==2 {print $4/1024/1024}' | cut -d. -f1 || echo "0"
    else
        echo "0"
    fi
}

###############################################################################
# Network Functions (Enhanced with timeout and retry)
###############################################################################

check_url() {
    local url="$1"
    local timeout="${2:-${DEFAULT_TIMEOUT}}"
    local max_retries="${3:-${DEFAULT_RETRIES}}"
    
    if ! validate_url "${url}"; then
        return 1
    fi
    
    local retries=0
    while [ $retries -lt $max_retries ]; do
        if command -v curl >/dev/null 2>&1; then
            if curl -f -s --max-time "${timeout}" "${url}" >/dev/null 2>&1; then
                return 0
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget --timeout="${timeout}" --tries=1 --spider "${url}" >/dev/null 2>&1; then
                return 0
            fi
        else
            log_error "Neither curl nor wget available for URL check"
            return 1
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $max_retries ]; then
            sleep 1
        fi
    done
    
    return 1
}

get_response_time() {
    local url="$1"
    local timeout="${2:-${DEFAULT_TIMEOUT}}"
    
    if ! validate_url "${url}"; then
        echo "0"
        return 1
    fi
    
    if command -v curl >/dev/null 2>&1; then
        curl -s -o /dev/null -w "%{time_total}" \
            --max-time "${timeout}" "${url}" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

get_http_status() {
    local url="$1"
    local timeout="${2:-${DEFAULT_TIMEOUT}}"
    
    if ! validate_url "${url}"; then
        echo "000"
        return 1
    fi
    
    if command -v curl >/dev/null 2>&1; then
        curl -s -o /dev/null -w "%{http_code}" \
            --max-time "${timeout}" "${url}" 2>/dev/null || echo "000"
    else
        echo "000"
    fi
}

###############################################################################
# Retry and Error Handling (Enhanced)
###############################################################################

retry_command() {
    local max_attempts="${1:-${DEFAULT_RETRIES}}"
    local delay="${2:-${DEFAULT_RETRY_DELAY}}"
    shift 2
    local cmd="$*"
    
    if [ -z "${cmd}" ]; then
        log_error "No command provided to retry"
        return 1
    fi
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        log_debug "Attempt ${attempt}/${max_attempts}: ${cmd}"
        if eval "${cmd}" >/dev/null 2>&1; then
            log_debug "Command succeeded on attempt ${attempt}"
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            log_debug "Command failed, retrying in ${delay}s"
            sleep "${delay}"
        fi
        
        attempt=$((attempt + 1))
    done
    
    log_error "Command failed after ${max_attempts} attempts: ${cmd}"
    return 1
}

# Safe command execution with timeout
safe_execute() {
    local timeout="${1:-${DEFAULT_TIMEOUT}}"
    shift
    local cmd="$*"
    
    if command -v timeout >/dev/null 2>&1; then
        timeout "${timeout}" bash -c "${cmd}" 2>&1
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "${timeout}" bash -c "${cmd}" 2>&1
    else
        # Fallback: execute without timeout
        eval "${cmd}" 2>&1
    fi
}

###############################################################################
# File and Directory Functions (Enhanced)
###############################################################################

create_backup_marker() {
    local backup_path="$1"
    local marker_file="${backup_path}/.backup_marker"
    
    if ! check_directory_exists "${backup_path}"; then
        mkdir -p "${backup_path}" || return 1
    fi
    
    cat > "${marker_file}" <<EOF
BACKUP_DATE=$(date -Iseconds 2>/dev/null || date)
BACKUP_HOSTNAME=$(hostname 2>/dev/null || echo "unknown")
BACKUP_USER=$(whoami 2>/dev/null || echo "unknown")
BACKUP_VERSION=${SCRIPT_VERSION}
BACKUP_OS=$(uname -s 2>/dev/null || echo "unknown")
BACKUP_ARCH=$(uname -m 2>/dev/null || echo "unknown")
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
    
    if [ -z "$(ls -A "${backup_path}" 2>/dev/null)" ]; then
        log_error "Backup directory is empty"
        return 1
    fi
    
    log_debug "Backup integrity verified"
    return 0
}

calculate_directory_size() {
    local dir_path="$1"
    if command -v du >/dev/null 2>&1; then
        du -sh "${dir_path}" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Safe file copy with verification
safe_copy() {
    local source="$1"
    local destination="$2"
    
    if ! check_file_exists "${source}"; then
        return 1
    fi
    
    local dest_dir
    dest_dir=$(dirname "${destination}")
    if [ ! -d "${dest_dir}" ]; then
        mkdir -p "${dest_dir}" || return 1
    fi
    
    if cp "${source}" "${destination}" 2>/dev/null; then
        # Verify copy
        if [ -f "${destination}" ]; then
            log_debug "File copied successfully: ${destination}"
            return 0
        fi
    fi
    
    log_error "Failed to copy file: ${source} -> ${destination}"
    return 1
}

###############################################################################
# Utility Functions (Enhanced)
###############################################################################

generate_report() {
    local report_file="$1"
    local title="$2"
    shift 2
    local content="$*"
    
    local report_dir
    report_dir=$(dirname "${report_file}")
    if [ ! -d "${report_dir}" ]; then
        mkdir -p "${report_dir}" || return 1
    fi
    
    cat > "${report_file}" <<EOF
========================================
${title}
========================================
Generated: $(date -Iseconds 2>/dev/null || date)
Hostname: $(hostname 2>/dev/null || echo "unknown")
User: $(whoami 2>/dev/null || echo "unknown")
Script Version: ${SCRIPT_VERSION}

${content}

========================================
EOF
    
    log_debug "Report generated: ${report_file}"
    echo "${report_file}"
}

format_duration() {
    local seconds="${1:-0}"
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

# Parse command line arguments with getopts
parse_args() {
    local options="$1"
    shift
    local long_options="${1:-}"
    shift
    
    OPTIND=1
    while getopts "${options}" opt; do
        case $opt in
            h|help)
                return 2  # Special return code for help
                ;;
            v|version)
                echo "Version: ${SCRIPT_VERSION}"
                return 2
                ;;
            d|debug)
                export DEBUG=true
                export VERBOSE=true
                ;;
            *)
                log_error "Invalid option: -$opt"
                return 1
                ;;
        esac
    done
    
    return 0
}

###############################################################################
# Export Functions
###############################################################################

# Export all functions for use in other scripts
export -f log_info log_warn log_error log_debug log_success log_section
export -f init_logging
export -f check_command check_file_exists check_directory_exists check_disk_space
export -f validate_url validate_email validate_port validate_number
export -f check_aws_credentials send_sns_alert send_cloudwatch_metric upload_to_s3
export -f get_cpu_usage get_memory_usage get_disk_usage get_disk_available
export -f check_url get_response_time get_http_status
export -f create_backup_marker verify_backup_integrity calculate_directory_size safe_copy
export -f is_process_running get_process_count
export -f check_docker_container get_docker_container_status get_docker_container_health
export -f generate_report format_duration retry_command safe_execute parse_args

