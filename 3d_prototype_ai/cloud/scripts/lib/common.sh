#!/bin/bash
# Common utility functions for deployment scripts
# This library provides reusable functions following DevOps best practices

set -o errexit
set -o nounset
set -o pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
    fi
}

# Error handling
error_exit() {
    local exit_code="${1:-1}"
    local message="${2:-Unknown error}"
    log_error "${message}"
    exit "${exit_code}"
}

# Cleanup function for trap
cleanup() {
    local exit_code=$?
    log_debug "Cleaning up temporary files..."
    # Remove temporary files if they exist
    [ -n "${TMP_DIR:-}" ] && [ -d "${TMP_DIR}" ] && rm -rf "${TMP_DIR}"
    
    if [ ${exit_code} -ne 0 ]; then
        log_error "Script exited with code ${exit_code}"
    fi
    
    exit ${exit_code}
}

# Set up trap for cleanup
setup_trap() {
    trap cleanup EXIT INT TERM
}

# Create temporary directory
create_temp_dir() {
    TMP_DIR=$(mktemp -d -t deploy-XXXXXX)
    export TMP_DIR
    log_debug "Created temporary directory: ${TMP_DIR}"
}

# Validate required command exists
check_command() {
    local cmd="${1}"
    local package="${2:-${cmd}}"
    
    if ! command -v "${cmd}" &> /dev/null; then
        log_error "Command '${cmd}' not found. Please install ${package}"
        return 1
    fi
    log_debug "Command '${cmd}' found"
    return 0
}

# Validate AWS credentials
check_aws_credentials() {
    log_info "Validating AWS credentials..."
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        log_info "Run 'aws configure' to set up your credentials"
        return 1
    fi
    
    local aws_account
    aws_account=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
    log_info "AWS credentials valid (Account: ${aws_account})"
    return 0
}

# Validate file exists and is readable
validate_file() {
    local file_path="${1}"
    local description="${2:-File}"
    
    if [ ! -f "${file_path}" ]; then
        log_error "${description} not found: ${file_path}"
        return 1
    fi
    
    if [ ! -r "${file_path}" ]; then
        log_error "${description} is not readable: ${file_path}"
        return 1
    fi
    
    log_debug "Validated ${description}: ${file_path}"
    return 0
}

# Validate directory exists
validate_directory() {
    local dir_path="${1}"
    local description="${2:-Directory}"
    
    if [ ! -d "${dir_path}" ]; then
        log_error "${description} not found: ${dir_path}"
        return 1
    fi
    
    log_debug "Validated ${description}: ${dir_path}"
    return 0
}

# Validate IP address format
validate_ip() {
    local ip="${1}"
    
    if [[ ! ${ip} =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        log_error "Invalid IP address format: ${ip}"
        return 1
    fi
    
    log_debug "Validated IP address: ${ip}"
    return 0
}

# Validate port number
validate_port() {
    local port="${1}"
    
    if ! [[ "${port}" =~ ^[0-9]+$ ]] || [ "${port}" -lt 1 ] || [ "${port}" -gt 65535 ]; then
        log_error "Invalid port number: ${port} (must be 1-65535)"
        return 1
    fi
    
    log_debug "Validated port: ${port}"
    return 0
}

# Wait for service to be ready
wait_for_service() {
    local url="${1}"
    local max_attempts="${2:-30}"
    local delay="${3:-5}"
    local description="${4:-Service}"
    
    log_info "Waiting for ${description} to be ready..."
    
    local attempt=0
    while [ ${attempt} -lt ${max_attempts} ]; do
        if curl -sf "${url}" > /dev/null 2>&1; then
            log_info "${description} is ready ✓"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log_debug "Waiting... (${attempt}/${max_attempts})"
        sleep ${delay}
    done
    
    log_error "${description} did not become ready after ${max_attempts} attempts"
    return 1
}

# Retry command with exponential backoff
retry_with_backoff() {
    local max_attempts="${1}"
    local delay="${2}"
    shift 2
    local cmd=("$@")
    
    local attempt=0
    while [ ${attempt} -lt ${max_attempts} ]; do
        if "${cmd[@]}"; then
            return 0
        fi
        
        attempt=$((attempt + 1))
        if [ ${attempt} -lt ${max_attempts} ]; then
            log_warn "Command failed, retrying in ${delay}s (attempt ${attempt}/${max_attempts})..."
            sleep ${delay}
            delay=$((delay * 2)) # Exponential backoff
        fi
    done
    
    log_error "Command failed after ${max_attempts} attempts"
    return 1
}

# Get script directory
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "${source}" ]; do
        local dir
        dir="$(cd -P "$(dirname "${source}")" && pwd)"
        source="$(readlink "${source}")"
        [[ ${source} != /* ]] && source="${dir}/${source}"
    done
    echo "$(cd -P "$(dirname "${source}")" && pwd)"
}

# Load environment file
load_env_file() {
    local env_file="${1}"
    
    if [ ! -f "${env_file}" ]; then
        log_warn "Environment file not found: ${env_file}"
        return 1
    fi
    
    # Source the file, but don't fail if it doesn't exist
    set -a
    source "${env_file}" 2>/dev/null || true
    set +a
    
    log_debug "Loaded environment file: ${env_file}"
    return 0
}

# Validate required environment variables
validate_env_vars() {
    local missing_vars=()
    
    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("${var}")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    log_debug "All required environment variables are set"
    return 0
}

# Create backup
create_backup() {
    local source="${1}"
    local backup_dir="${2:-./backups}"
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="${backup_dir}/backup_${timestamp}.tar.gz"
    
    mkdir -p "${backup_dir}"
    
    log_info "Creating backup: ${backup_file}"
    tar -czf "${backup_file}" -C "$(dirname "${source}")" "$(basename "${source}")" 2>/dev/null || {
        log_error "Failed to create backup"
        return 1
    }
    
    log_info "Backup created: ${backup_file}"
    echo "${backup_file}"
}

# Check disk space
check_disk_space() {
    local path="${1:-/}"
    local required_gb="${2:-5}"
    
    local available_gb
    available_gb=$(df -BG "${path}" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "${available_gb}" -lt "${required_gb}" ]; then
        log_error "Insufficient disk space: ${available_gb}GB available, ${required_gb}GB required"
        return 1
    fi
    
    log_debug "Disk space check passed: ${available_gb}GB available"
    return 0
}

# Send notification (placeholder for integration with notification services)
send_notification() {
    local message="${1}"
    local level="${2:-info}"
    
    log_debug "Notification [${level}]: ${message}"
    # TODO: Integrate with notification services (Slack, email, etc.)
}

