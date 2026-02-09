#!/bin/bash
# Integration test script
# Tests integration between components

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

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run integration tests.

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -v, --verbose            Verbose output
    -h, --help               Show this help message

EXAMPLES:
    $0 --ip 1.2.3.4
    $0 --ip 1.2.3.4 --verbose

EOF
}

# Parse arguments
parse_args() {
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
            -v|--verbose)
                VERBOSE=true
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

# Test application endpoint
test_application() {
    local ip="${1}"
    local verbose="${2}"
    
    log_info "Testing application endpoint..."
    
    local health_status
    health_status=$(curl -sf -m 10 "http://${ip}:8030/health" 2>/dev/null && echo "pass" || echo "fail")
    
    if [ "${health_status}" = "pass" ]; then
        log_info "✓ Application health check passed"
        return 0
    else
        log_error "✗ Application health check failed"
        return 1
    fi
}

# Test SSH connectivity
test_ssh() {
    local ip="${1}"
    local key_path="${2}"
    local verbose="${3}"
    
    log_info "Testing SSH connectivity..."
    
    if ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 \
        ubuntu@${ip} \
        "echo 'SSH test successful'" > /dev/null 2>&1; then
        log_info "✓ SSH connectivity test passed"
        return 0
    else
        log_error "✗ SSH connectivity test failed"
        return 1
    fi
}

# Test Docker
test_docker() {
    local ip="${1}"
    local key_path="${2}"
    local verbose="${3}"
    
    log_info "Testing Docker..."
    
    local docker_status
    docker_status=$(ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} \
        "docker ps > /dev/null 2>&1 && echo 'pass' || echo 'fail'" 2>/dev/null || echo "fail")
    
    if [ "${docker_status}" = "pass" ]; then
        log_info "✓ Docker test passed"
        return 0
    else
        log_warn "⚠ Docker test failed or not available"
        return 1
    fi
}

# Test AWS connectivity
test_aws() {
    local verbose="${1}"
    
    log_info "Testing AWS connectivity..."
    
    if aws sts get-caller-identity > /dev/null 2>&1; then
        log_info "✓ AWS connectivity test passed"
        return 0
    else
        log_warn "⚠ AWS connectivity test failed (may not be configured)"
        return 1
    fi
}

# Run all integration tests
run_tests() {
    local ip="${1}"
    local key_path="${2}"
    local verbose="${3}"
    
    local tests_passed=0
    local tests_failed=0
    
    log_info "Running integration tests..."
    echo ""
    
    # Test application
    if test_application "${ip}" "${verbose}"; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # Test SSH
    if test_ssh "${ip}" "${key_path}" "${verbose}"; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # Test Docker
    if test_docker "${ip}" "${key_path}" "${verbose}"; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # Test AWS
    if test_aws "${verbose}"; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    echo ""
    log_info "=========================================="
    log_info "Integration Test Results"
    log_info "=========================================="
    log_info "Tests passed: ${tests_passed}"
    log_info "Tests failed: ${tests_failed}"
    log_info "Total tests: $((tests_passed + tests_failed))"
    echo ""
    
    if [ ${tests_failed} -eq 0 ]; then
        log_info "All integration tests passed! ✓"
        return 0
    else
        log_error "Some integration tests failed ✗"
        return 1
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${INSTANCE_IP}" ] || [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    validate_ip "${INSTANCE_IP}"
    validate_file "${AWS_KEY_PATH}" "SSH private key"
    
    run_tests "${INSTANCE_IP}" "${AWS_KEY_PATH}" "${VERBOSE}"
}

main "$@"


