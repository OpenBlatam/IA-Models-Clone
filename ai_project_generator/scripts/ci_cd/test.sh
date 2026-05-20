#!/bin/bash

###############################################################################
# CI/CD Test Script
# Automated testing script for CI/CD pipelines
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/../aws/scripts/common_functions_enhanced.sh" 2>/dev/null || {
    source "${SCRIPT_DIR}/../aws/scripts/common_functions.sh" 2>/dev/null || {
        echo "Error: common functions not found" >&2
        exit 1
    }
}

# Configuration
TEST_TYPE="${TEST_TYPE:-all}" # unit, integration, e2e, all
COVERAGE="${COVERAGE:-true}"
PARALLEL="${PARALLEL:-true}"
VERBOSE="${VERBOSE:-false}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

###############################################################################
# Test Functions
###############################################################################

run_unit_tests() {
    log_section "Running Unit Tests"
    
    cd "${PROJECT_ROOT}" || exit 1
    
    local pytest_args=(
        "tests/unit"
        "-v"
    )
    
    if [ "${COVERAGE}" = "true" ]; then
        pytest_args+=(
            "--cov=."
            "--cov-report=xml"
            "--cov-report=html"
            "--cov-report=term"
        )
    fi
    
    if [ "${PARALLEL}" = "true" ]; then
        pytest_args+=("-n" "auto")
    fi
    
    if [ "${VERBOSE}" = "true" ]; then
        pytest_args+=("-vv")
    fi
    
    log_info "Running unit tests..."
    if pytest "${pytest_args[@]}"; then
        log_success "Unit tests passed"
        return 0
    else
        log_error "Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    log_section "Running Integration Tests"
    
    cd "${PROJECT_ROOT}" || exit 1
    
    # Check if integration tests exist
    if [ ! -d "tests/integration" ]; then
        log_warn "Integration tests directory not found, skipping"
        return 0
    fi
    
    local pytest_args=(
        "tests/integration"
        "-v"
    )
    
    if [ "${VERBOSE}" = "true" ]; then
        pytest_args+=("-vv")
    fi
    
    # Set up test environment
    export TEST_ENV=true
    export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
    
    log_info "Running integration tests..."
    if pytest "${pytest_args[@]}"; then
        log_success "Integration tests passed"
        return 0
    else
        log_error "Integration tests failed"
        return 1
    fi
}

run_e2e_tests() {
    log_section "Running End-to-End Tests"
    
    cd "${PROJECT_ROOT}" || exit 1
    
    # Check if E2E tests exist
    if [ ! -d "tests/e2e" ]; then
        log_warn "E2E tests directory not found, skipping"
        return 0
    fi
    
    # Start application for E2E tests
    log_info "Starting application for E2E tests..."
    # Add application startup logic here
    
    local pytest_args=(
        "tests/e2e"
        "-v"
    )
    
    if [ "${VERBOSE}" = "true" ]; then
        pytest_args+=("-vv")
    fi
    
    log_info "Running E2E tests..."
    if pytest "${pytest_args[@]}"; then
        log_success "E2E tests passed"
        return 0
    else
        log_error "E2E tests failed"
        return 1
    fi
}

run_linting() {
    log_section "Running Linting"
    
    cd "${PROJECT_ROOT}" || exit 1
    
    local lint_errors=0
    
    # Run Black (format check)
    if check_command "black"; then
        log_info "Running Black format check..."
        if black --check --diff .; then
            log_success "Black check passed"
        else
            log_warn "Black check found formatting issues"
            lint_errors=$((lint_errors + 1))
        fi
    fi
    
    # Run Ruff
    if check_command "ruff"; then
        log_info "Running Ruff linting..."
        if ruff check .; then
            log_success "Ruff check passed"
        else
            log_warn "Ruff check found issues"
            lint_errors=$((lint_errors + 1))
        fi
    fi
    
    # Run MyPy
    if check_command "mypy"; then
        log_info "Running MyPy type checking..."
        if mypy . --ignore-missing-imports; then
            log_success "MyPy check passed"
        else
            log_warn "MyPy check found type issues"
            lint_errors=$((lint_errors + 1))
        fi
    fi
    
    if [ $lint_errors -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

run_security_tests() {
    log_section "Running Security Tests"
    
    cd "${PROJECT_ROOT}" || exit 1
    
    local security_errors=0
    
    # Run Bandit
    if check_command "bandit"; then
        log_info "Running Bandit security scan..."
        if bandit -r . -f json -o bandit-report.json; then
            log_success "Bandit scan passed"
        else
            log_warn "Bandit scan found security issues"
            security_errors=$((security_errors + 1))
        fi
    fi
    
    # Run Safety
    if check_command "safety"; then
        log_info "Running Safety dependency check..."
        if safety check --json; then
            log_success "Safety check passed"
        else
            log_warn "Safety check found vulnerable dependencies"
            security_errors=$((security_errors + 1))
        fi
    fi
    
    if [ $security_errors -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

###############################################################################
# Main Execution
###############################################################################

main() {
    init_logging "/var/log/cicd-test.log"
    log_section "CI/CD Test Process"
    log_info "Test Type: ${TEST_TYPE}"
    log_info "Coverage: ${COVERAGE}"
    log_info "Parallel: ${PARALLEL}"
    
    local exit_code=0
    
    case "${TEST_TYPE}" in
        unit)
            run_unit_tests || exit_code=1
            ;;
        integration)
            run_integration_tests || exit_code=1
            ;;
        e2e)
            run_e2e_tests || exit_code=1
            ;;
        lint)
            run_linting || exit_code=1
            ;;
        security)
            run_security_tests || exit_code=1
            ;;
        all)
            run_linting || exit_code=1
            run_security_tests || exit_code=1
            run_unit_tests || exit_code=1
            run_integration_tests || exit_code=1
            run_e2e_tests || exit_code=1
            ;;
        *)
            log_error "Unknown test type: ${TEST_TYPE}"
            exit 1
            ;;
    esac
    
    if [ $exit_code -eq 0 ]; then
        log_success "Test process completed successfully"
    else
        log_error "Test process failed"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"

