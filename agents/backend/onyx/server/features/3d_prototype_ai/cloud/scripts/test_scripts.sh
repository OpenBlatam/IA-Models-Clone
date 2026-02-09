#!/bin/bash
# Test script for deployment scripts
# Validates script syntax and basic functionality

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."
TESTS_PASSED=0
TESTS_FAILED=0

# Test script syntax
test_script_syntax() {
    local script_path="${1}"
    local script_name="${2}"
    
    log_info "Testing syntax: ${script_name}"
    
    if bash -n "${script_path}" 2>/dev/null; then
        log_info "✓ ${script_name} syntax is valid"
        ((TESTS_PASSED++))
        return 0
    else
        log_error "✗ ${script_name} has syntax errors"
        bash -n "${script_path}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test shellcheck if available
test_shellcheck() {
    local script_path="${1}"
    local script_name="${2}"
    
    if ! command -v shellcheck &> /dev/null; then
        log_warn "shellcheck not installed, skipping lint check"
        return 0
    fi
    
    log_info "Running shellcheck: ${script_name}"
    
    if shellcheck "${script_path}" > /dev/null 2>&1; then
        log_info "✓ ${script_name} passed shellcheck"
        ((TESTS_PASSED++))
        return 0
    else
        log_warn "✗ ${script_name} has shellcheck warnings"
        shellcheck "${script_path}" || true
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test common library functions
test_common_library() {
    log_info "Testing common library functions..."
    
    local test_file="${TMP_DIR}/test_common.sh"
    
    cat > "${test_file}" << 'EOF'
#!/bin/bash
source lib/common.sh

# Test logging functions
log_info "Test info message"
log_warn "Test warn message"
log_error "Test error message"
log_debug "Test debug message"

# Test validation functions
validate_ip "127.0.0.1" || exit 1
validate_port "8080" || exit 1
validate_file "lib/common.sh" "Common library" || exit 1

echo "All common library tests passed"
EOF
    
    chmod +x "${test_file}"
    cd "${SCRIPT_DIR}"
    
    if bash "${test_file}" > /dev/null 2>&1; then
        log_info "✓ Common library functions work correctly"
        ((TESTS_PASSED++))
        return 0
    else
        log_error "✗ Common library functions have issues"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test script argument parsing
test_script_args() {
    local script_path="${1}"
    local script_name="${2}"
    
    log_info "Testing argument parsing: ${script_name}"
    
    # Test help flag
    if "${script_path}" --help > /dev/null 2>&1 || "${script_path}" -h > /dev/null 2>&1; then
        log_info "✓ ${script_name} help works"
        ((TESTS_PASSED++))
    else
        log_warn "✗ ${script_name} help may not work"
        ((TESTS_FAILED++))
    fi
}

# Main test function
main() {
    log_info "Starting script tests..."
    echo ""
    
    setup_trap
    create_temp_dir
    
    # Test common library
    test_common_library
    echo ""
    
    # Test all scripts
    local scripts=(
        "deploy.sh:Deploy script"
        "validate.sh:Validate script"
        "backup.sh:Backup script"
        "rollback.sh:Rollback script"
        "health_check.sh:Health check script"
        "monitor.sh:Monitor script"
        "update_app.sh:Update script"
        "view_logs.sh:View logs script"
        "launch_ec2.sh:Launch EC2 script"
    )
    
    for script_info in "${scripts[@]}"; do
        local script_file
        script_file=$(echo "${script_info}" | cut -d':' -f1)
        local script_name
        script_name=$(echo "${script_info}" | cut -d':' -f2)
        local script_path="${SCRIPT_DIR}/${script_file}"
        
        if [ ! -f "${script_path}" ]; then
            log_warn "Script not found: ${script_file}, skipping..."
            continue
        fi
        
        test_script_syntax "${script_path}" "${script_name}"
        test_shellcheck "${script_path}" "${script_name}"
        
        # Test argument parsing for scripts that support it
        if [[ "${script_file}" =~ ^(deploy|backup|rollback|monitor|health_check)\.sh$ ]]; then
            test_script_args "${script_path}" "${script_name}"
        fi
        
        echo ""
    done
    
    # Summary
    log_info "=========================================="
    log_info "Test Summary"
    log_info "=========================================="
    log_info "Tests passed: ${TESTS_PASSED}"
    log_info "Tests failed: ${TESTS_FAILED}"
    log_info "Total tests: $((TESTS_PASSED + TESTS_FAILED))"
    echo ""
    
    if [ ${TESTS_FAILED} -eq 0 ]; then
        log_info "All tests passed! ✓"
        exit 0
    else
        log_error "Some tests failed ✗"
        exit 1
    fi
}

main "$@"

