#!/bin/bash
# Common Library for Deployment Scripts
# Reusable functions following DevOps best practices

set -euo pipefail

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_debug() {
    if [ "${DEBUG:-false}" == "true" ]; then
        echo -e "${CYAN}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
    fi
}

# Error handling
setup_error_handling() {
    set -euo pipefail
    
    trap 'cleanup_on_error $? $LINENO' ERR
    trap 'cleanup_on_exit' EXIT
    
    readonly TEMP_FILES=()
}

cleanup_on_error() {
    local exit_code=$1
    local line_number=$2
    
    log_error "Script failed at line ${line_number} with exit code ${exit_code}"
    cleanup_on_exit
    exit "${exit_code}"
}

cleanup_on_exit() {
    log_debug "Cleaning up temporary files..."
    for temp_file in "${TEMP_FILES[@]}"; do
        if [ -f "${temp_file}" ]; then
            rm -f "${temp_file}"
            log_debug "Removed temporary file: ${temp_file}"
        fi
    done
}

register_temp_file() {
    local file_path="$1"
    TEMP_FILES+=("${file_path}")
}

# Validation functions
validate_command() {
    local command="$1"
    local package="${2:-${command}}"
    
    if ! command -v "${command}" &> /dev/null; then
        log_error "Command '${command}' is not installed"
        log_info "Please install ${package}"
        return 1
    fi
    return 0
}

validate_required_vars() {
    local missing_vars=()
    
    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("${var}")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    return 0
}

validate_file_exists() {
    local file_path="$1"
    
    if [ ! -f "${file_path}" ]; then
        log_error "File not found: ${file_path}"
        return 1
    fi
    return 0
}

validate_directory_exists() {
    local dir_path="$1"
    
    if [ ! -d "${dir_path}" ]; then
        log_error "Directory not found: ${dir_path}"
        return 1
    fi
    return 0
}

# Retry function with exponential backoff
retry_with_backoff() {
    local max_attempts="${1:-3}"
    local base_delay="${2:-2}"
    local command="${@:3}"
    local attempt=1
    local delay="${base_delay}"
    
    while [ ${attempt} -le ${max_attempts} ]; do
        log_debug "Attempt ${attempt}/${max_attempts}: ${command}"
        
        if eval "${command}"; then
            return 0
        fi
        
        if [ ${attempt} -lt ${max_attempts} ]; then
            log_warn "Command failed, retrying in ${delay}s..."
            sleep ${delay}
            delay=$((delay * 2))
        fi
        
        attempt=$((attempt + 1))
    done
    
    log_error "Command failed after ${max_attempts} attempts"
    return 1
}

# Wait for condition
wait_for_condition() {
    local condition_command="$1"
    local max_wait="${2:-300}"
    local interval="${3:-5}"
    local elapsed=0
    
    log_info "Waiting for condition: ${condition_command}"
    
    while [ ${elapsed} -lt ${max_wait} ]; do
        if eval "${condition_command}"; then
            log_success "Condition met"
            return 0
        fi
        
        sleep ${interval}
        elapsed=$((elapsed + interval))
        log_debug "Waiting... (${elapsed}/${max_wait}s)"
    done
    
    log_error "Condition not met within ${max_wait}s"
    return 1
}

# Health check
health_check() {
    local url="${1:-}"
    local max_attempts="${2:-10}"
    local delay="${3:-5}"
    
    if [ -z "${url}" ]; then
        log_error "Health check URL not provided"
        return 1
    fi
    
    log_info "Performing health check on ${url}"
    
    local attempt=1
    while [ ${attempt} -le ${max_attempts} ]; do
        if curl -f -s -m 10 "${url}" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        if [ ${attempt} -lt ${max_attempts} ]; then
            log_warn "Health check attempt ${attempt}/${max_attempts} failed, retrying in ${delay}s..."
            sleep ${delay}
        fi
        
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after ${max_attempts} attempts"
    return 1
}

# Get script directory
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "${source}" ]; do
        local dir="$(cd -P "$(dirname "${source}")" && pwd)"
        source="$(readlink "${source}")"
        [[ ${source} != /* ]] && source="${dir}/${source}"
    done
    echo "$(cd -P "$(dirname "${source}")" && pwd)"
}

# Get project root
get_project_root() {
    local script_dir=$(get_script_dir)
    echo "$(cd "${script_dir}/../../.." && pwd)"
}

# Create temporary file
create_temp_file() {
    local prefix="${1:-tmp}"
    local temp_file=$(mktemp "/tmp/${prefix}.XXXXXX")
    register_temp_file "${temp_file}"
    echo "${temp_file}"
}

# Create temporary directory
create_temp_dir() {
    local prefix="${1:-tmp}"
    local temp_dir=$(mktemp -d "/tmp/${prefix}.XXXXXX")
    register_temp_file "${temp_dir}"
    echo "${temp_dir}"
}

# Parse command line arguments
parse_args() {
    local args=("$@")
    local i=0
    
    while [ $i -lt ${#args[@]} ]; do
        case "${args[$i]}" in
            --*)
                local key="${args[$i]#--}"
                local value="${args[$i+1]:-}"
                
                # Handle boolean flags
                if [[ "${value}" == --* ]] || [ -z "${value}" ]; then
                    export "${key//-/_}=true"
                else
                    export "${key//-/_}=${value}"
                    i=$((i + 1))
                fi
                ;;
            -*)
                local key="${args[$i]#-}"
                local value="${args[$i+1]:-}"
                export "${key}=${value}"
                i=$((i + 1))
                ;;
            *)
                # Positional argument
                export "ARG_${i}=${args[$i]}"
                ;;
        esac
        i=$((i + 1))
    done
}

# Check if running in CI
is_ci() {
    [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ] || [ -n "${AZURE_DEVOPS:-}" ]
}

# Get environment
get_environment() {
    echo "${ENVIRONMENT:-${CI_ENVIRONMENT_NAME:-production}}"
}

# Generate unique ID
generate_id() {
    date +%s%N | sha256sum | head -c 16
}

# Format bytes
format_bytes() {
    local bytes="$1"
    local units=("B" "KB" "MB" "GB" "TB")
    local unit_index=0
    
    while (( $(echo "${bytes} > 1024" | bc -l) )) && [ ${unit_index} -lt $((${#units[@]} - 1)) ]; do
        bytes=$(echo "scale=2; ${bytes} / 1024" | bc)
        unit_index=$((unit_index + 1))
    done
    
    echo "${bytes} ${units[${unit_index}]}"
}

# Check port availability
check_port() {
    local port="$1"
    
    if command -v netstat &> /dev/null; then
        netstat -tuln | grep -q ":${port} " && return 1 || return 0
    elif command -v ss &> /dev/null; then
        ss -tuln | grep -q ":${port} " && return 1 || return 0
    else
        log_warn "Cannot check port availability (netstat/ss not available)"
        return 0
    fi
}

# Initialize common library
init_common() {
    setup_error_handling
    log_debug "Common library initialized"
}

# Export functions
export -f log_info log_success log_error log_warn log_debug
export -f validate_command validate_required_vars validate_file_exists validate_directory_exists
export -f retry_with_backoff wait_for_condition health_check
export -f get_script_dir get_project_root create_temp_file create_temp_dir
export -f parse_args is_ci get_environment generate_id format_bytes check_port




