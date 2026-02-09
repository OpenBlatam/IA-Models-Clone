#!/bin/bash
# Dependency update script
# Checks and updates application dependencies

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Check and update application dependencies.

OPTIONS:
    -c, --check-only         Only check, don't update
    -u, --update            Update dependencies
    -s, --security           Check for security vulnerabilities
    -o, --output FILE       Output file for report
    -h, --help              Show this help message

EXAMPLES:
    $0 --check-only
    $0 --update
    $0 --security --output security_report.json

EOF
}

# Parse arguments
parse_args() {
    CHECK_ONLY=false
    UPDATE=false
    SECURITY=false
    OUTPUT_FILE=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--check-only)
                CHECK_ONLY=true
                shift
                ;;
            -u|--update)
                UPDATE=true
                shift
                ;;
            -s|--security)
                SECURITY=true
                shift
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
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

# Check for outdated packages
check_outdated() {
    local project_path="${1}"
    
    cd "${project_path}"
    
    log_info "Checking for outdated packages..."
    
    if [ -f "requirements.txt" ]; then
        pip list --outdated 2>/dev/null || true
    else
        log_warn "requirements.txt not found"
    fi
}

# Update dependencies
update_dependencies() {
    local project_path="${1}"
    
    cd "${project_path}"
    
    if [ ! -f "requirements.txt" ]; then
        error_exit 1 "requirements.txt not found"
    fi
    
    log_info "Updating dependencies..."
    
    # Create backup
    cp requirements.txt requirements.txt.backup
    
    # Update pip
    pip install --upgrade pip
    
    # Update packages
    pip install --upgrade -r requirements.txt
    
    # Generate new requirements
    pip freeze > requirements.txt.new
    
    log_info "Updated requirements saved to requirements.txt.new"
    log_info "Original requirements backed up to requirements.txt.backup"
    log_info "Review changes before applying!"
}

# Check security vulnerabilities
check_security() {
    local project_path="${1}"
    local output_file="${2}"
    
    cd "${project_path}"
    
    log_info "Checking for security vulnerabilities..."
    
    # Install safety if not available
    if ! command -v safety &> /dev/null; then
        pip install safety
    fi
    
    if [ -n "${output_file}" ]; then
        safety check --json --output "${output_file}" || true
        log_info "Security report saved to: ${output_file}"
    else
        safety check || true
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ "${CHECK_ONLY}" = "true" ]; then
        check_outdated "${PROJECT_ROOT}"
    elif [ "${UPDATE}" = "true" ]; then
        update_dependencies "${PROJECT_ROOT}"
    elif [ "${SECURITY}" = "true" ]; then
        check_security "${PROJECT_ROOT}" "${OUTPUT_FILE}"
    else
        log_info "Running all checks..."
        check_outdated "${PROJECT_ROOT}"
        echo ""
        check_security "${PROJECT_ROOT}" "${OUTPUT_FILE}"
    fi
}

main "$@"


