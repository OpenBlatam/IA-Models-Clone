#!/bin/bash
# CI Helper Script
# Utility functions for CI/CD workflows

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if running in CI
is_ci() {
    [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]
}

# Get changed files
get_changed_files() {
    if [ -n "${GITHUB_BASE_REF:-}" ]; then
        # Pull request
        git diff --name-only origin/${GITHUB_BASE_REF}...HEAD
    else
        # Push
        git diff --name-only HEAD~1 HEAD
    fi
}

# Check if path changed
path_changed() {
    local path="$1"
    get_changed_files | grep -q "^${path}" || return 1
}

# Run command with retry
retry() {
    local max_attempts="${1:-3}"
    local delay="${2:-5}"
    shift 2
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        fi

        if [ $attempt -lt $max_attempts ]; then
            log_warn "Attempt $attempt failed. Retrying in ${delay}s..."
            sleep $delay
        fi

        attempt=$((attempt + 1))
    done

    log_error "Command failed after $max_attempts attempts"
    return 1
}

# Wait for service
wait_for_service() {
    local url="$1"
    local max_attempts="${2:-30}"
    local delay="${3:-2}"
    local attempt=1

    log_info "Waiting for service at ${url}..."

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "${url}" > /dev/null 2>&1; then
            log_success "Service is ready!"
            return 0
        fi

        if [ $attempt -lt $max_attempts ]; then
            log_info "Attempt $attempt/$max_attempts - waiting ${delay}s..."
            sleep $delay
        fi

        attempt=$((attempt + 1))
    done

    log_error "Service did not become ready after $max_attempts attempts"
    return 1
}

# Get test coverage
get_coverage() {
    local coverage_file="${1:-coverage.xml}"
    
    if [ -f "${coverage_file}" ]; then
        python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('${coverage_file}')
root = tree.getroot()
coverage = root.get('line-rate', '0')
print(f'{float(coverage) * 100:.2f}%')
" 2>/dev/null || echo "N/A"
}

# Check code quality thresholds
check_quality_thresholds() {
    local min_coverage="${1:-80}"
    local coverage=$(get_coverage)
    
    if [ "${coverage}" != "N/A" ]; then
        local coverage_num=$(echo "${coverage}" | sed 's/%//')
        if (( $(echo "${coverage_num} < ${min_coverage}" | bc -l) )); then
            log_error "Coverage ${coverage} is below threshold ${min_coverage}%"
            return 1
        else
            log_success "Coverage ${coverage} meets threshold ${min_coverage}%"
        fi
    fi
}

# Generate summary
generate_summary() {
    local summary_file="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
    
    cat >> "${summary_file}" << EOF
## CI/CD Summary

### Test Results
- Coverage: $(get_coverage)
- Status: $([ $? -eq 0 ] && echo "✅ Passed" || echo "❌ Failed")

### Build Information
- Python Version: ${PYTHON_VERSION:-N/A}
- Docker Image: ${DOCKER_IMAGE_NAME:-N/A}
- Commit: ${GITHUB_SHA:-N/A}

EOF
}

# Main function
main() {
    case "${1:-}" in
        wait-for-service)
            wait_for_service "${2:-}" "${3:-30}" "${4:-2}"
            ;;
        check-quality)
            check_quality_thresholds "${2:-80}"
            ;;
        get-coverage)
            get_coverage "${2:-coverage.xml}"
            ;;
        generate-summary)
            generate_summary
            ;;
        *)
            echo "Usage: $0 {wait-for-service|check-quality|get-coverage|generate-summary}"
            exit 1
            ;;
    esac
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi




