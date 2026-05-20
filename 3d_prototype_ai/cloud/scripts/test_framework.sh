#!/bin/bash
# Advanced testing framework
# Comprehensive testing suite

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
readonly TEST_SUITE="${TEST_SUITE:-all}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Advanced testing framework.

COMMANDS:
    unit                Run unit tests
    integration         Run integration tests
    e2e                 Run end-to-end tests
    load                Run load tests
    security            Run security tests
    all                 Run all tests

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -s, --suite SUITE         Test suite (default: all)
    -v, --verbose            Verbose output
    -h, --help               Show this help message

EXAMPLES:
    $0 unit
    $0 integration --ip 1.2.3.4
    $0 all --verbose

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    VERBOSE=false
    
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
            -s|--suite)
                TEST_SUITE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            unit|integration|e2e|load|security|all)
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

# Run unit tests
run_unit_tests() {
    local verbose="${1}"
    
    log_info "Running unit tests..."
    
    local project_root="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    cd "${project_root}"
    
    local pytest_args="-v"
    if [ "${verbose}" = "true" ]; then
        pytest_args="${pytest_args} -s"
    fi
    
    pytest tests/unit/ ${pytest_args} --cov=. --cov-report=term || {
        log_error "Unit tests failed"
        return 1
    }
    
    log_info "Unit tests passed"
}

# Run integration tests
run_integration_tests() {
    local ip="${1}"
    local key_path="${2}"
    local verbose="${3}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Running integration tests..."
    
    ./scripts/integration_test.sh --ip "${ip}" --key-path "${key_path}" ${verbose:+-v} || {
        log_error "Integration tests failed"
        return 1
    }
    
    log_info "Integration tests passed"
}

# Run end-to-end tests
run_e2e_tests() {
    local ip="${1}"
    local verbose="${2}"
    
    if [ -z "${ip}" ]; then
        error_exit 1 "INSTANCE_IP is required"
    fi
    
    log_info "Running end-to-end tests..."
    
    # Test application endpoints
    local endpoints=("/health" "/api/status" "/api/version")
    local failed=0
    
    for endpoint in "${endpoints[@]}"; do
        if curl -sf "http://${ip}:8030${endpoint}" > /dev/null 2>&1; then
            log_info "✓ ${endpoint} - OK"
        else
            log_error "✗ ${endpoint} - Failed"
            failed=$((failed + 1))
        fi
    done
    
    if [ ${failed} -eq 0 ]; then
        log_info "End-to-end tests passed"
        return 0
    else
        log_error "End-to-end tests failed (${failed} failures)"
        return 1
    fi
}

# Run load tests
run_load_tests() {
    local ip="${1}"
    
    if [ -z "${ip}" ]; then
        error_exit 1 "INSTANCE_IP is required"
    fi
    
    log_info "Running load tests..."
    
    ./scripts/performance_test.sh \
        --url "http://${ip}:8030/health" \
        --concurrent 50 \
        --duration 300 || {
        log_error "Load tests failed"
        return 1
    }
    
    log_info "Load tests completed"
}

# Run security tests
run_security_tests() {
    log_info "Running security tests..."
    
    # Run security scans
    ./scripts/security_hardening.sh audit || {
        log_error "Security tests failed"
        return 1
    }
    
    log_info "Security tests passed"
}

# Run all tests
run_all_tests() {
    local ip="${1}"
    local key_path="${2}"
    local verbose="${3}"
    
    log_info "Running all tests..."
    echo ""
    
    local tests_passed=0
    local tests_failed=0
    
    # Unit tests
    if run_unit_tests "${verbose}"; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    echo ""
    
    # Integration tests
    if [ -n "${ip}" ] && [ -n "${key_path}" ]; then
        if run_integration_tests "${ip}" "${key_path}" "${verbose}"; then
            ((tests_passed++))
        else
            ((tests_failed++))
        fi
        echo ""
    fi
    
    # E2E tests
    if [ -n "${ip}" ]; then
        if run_e2e_tests "${ip}" "${verbose}"; then
            ((tests_passed++))
        else
            ((tests_failed++))
        fi
        echo ""
    fi
    
    # Security tests
    if run_security_tests; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    echo ""
    
    # Summary
    log_info "Test Summary:"
    log_info "  Passed: ${tests_passed}"
    log_info "  Failed: ${tests_failed}"
    echo ""
    
    if [ ${tests_failed} -eq 0 ]; then
        log_info "All tests passed! ✓"
        return 0
    else
        log_error "Some tests failed ✗"
        return 1
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        unit)
            run_unit_tests "${VERBOSE}"
            ;;
        integration)
            if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
                error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
            fi
            run_integration_tests "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${VERBOSE}"
            ;;
        e2e)
            if [ -z "${INSTANCE_IP}" ]; then
                error_exit 1 "INSTANCE_IP is required"
            fi
            run_e2e_tests "${INSTANCE_IP}" "${VERBOSE}"
            ;;
        load)
            if [ -z "${INSTANCE_IP}" ]; then
                error_exit 1 "INSTANCE_IP is required"
            fi
            run_load_tests "${INSTANCE_IP}"
            ;;
        security)
            run_security_tests
            ;;
        all)
            run_all_tests "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${VERBOSE}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


