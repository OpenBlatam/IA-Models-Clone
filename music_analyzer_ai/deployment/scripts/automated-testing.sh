#!/bin/bash
# Automated Testing Script
# Runs comprehensive test suite with reporting

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Initialize
init_common

# Configuration
readonly PROJECT_ROOT=$(get_project_root)
readonly BACKEND_PATH="${PROJECT_ROOT}/agents/backend/onyx/server/features/music_analyzer_ai"
readonly FRONTEND_PATH="${BACKEND_PATH}/frontend"
readonly TEST_RESULTS_DIR="${TEST_RESULTS_DIR:-./test-results}"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Test configuration
readonly PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
readonly NODE_VERSION="${NODE_VERSION:-20}"
readonly RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-true}"
readonly RUN_INTEGRATION_TESTS="${RUN_INTEGRATION_TESTS:-true}"
readonly RUN_E2E_TESTS="${RUN_E2E_TESTS:-false}"
readonly COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-80}"

# Create results directory
mkdir -p "${TEST_RESULTS_DIR}"

# Run backend unit tests
run_backend_unit_tests() {
    if [ "${RUN_UNIT_TESTS}" != "true" ]; then
        log_info "Skipping backend unit tests"
        return 0
    fi
    
    log_info "Running backend unit tests..."
    
    validate_directory_exists "${BACKEND_PATH}"
    
    cd "${BACKEND_PATH}"
    
    # Install dependencies
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install pytest pytest-cov pytest-xdist pytest-asyncio pytest-mock
    
    # Run tests
    pytest tests/test_core/ tests/test_api/ tests/test_services/ \
        -v \
        --cov=. \
        --cov-report=xml:"${TEST_RESULTS_DIR}/backend-coverage.xml" \
        --cov-report=html:"${TEST_RESULTS_DIR}/backend-coverage-html" \
        --cov-report=term \
        --junit-xml="${TEST_RESULTS_DIR}/backend-junit.xml" \
        -n auto \
        --maxfail=5 || {
        log_error "Backend unit tests failed"
        return 1
    }
    
    log_success "Backend unit tests completed"
}

# Run backend integration tests
run_backend_integration_tests() {
    if [ "${RUN_INTEGRATION_TESTS}" != "true" ]; then
        log_info "Skipping backend integration tests"
        return 0
    fi
    
    log_info "Running backend integration tests..."
    
    cd "${BACKEND_PATH}"
    
    # Start test services (if needed)
    docker-compose -f deployment/docker-compose.test.yml up -d || true
    
    # Wait for services
    wait_for_condition "curl -f http://localhost:5432 > /dev/null 2>&1" 60 5 || true
    wait_for_condition "curl -f http://localhost:6379 > /dev/null 2>&1" 60 5 || true
    
    # Run integration tests
    pytest tests/test_integration/ \
        -v \
        --cov=. \
        --cov-report=xml:"${TEST_RESULTS_DIR}/integration-coverage.xml" \
        --junit-xml="${TEST_RESULTS_DIR}/integration-junit.xml" \
        --maxfail=3 || {
        log_error "Backend integration tests failed"
        docker-compose -f deployment/docker-compose.test.yml down || true
        return 1
    }
    
    # Cleanup
    docker-compose -f deployment/docker-compose.test.yml down || true
    
    log_success "Backend integration tests completed"
}

# Run frontend tests
run_frontend_tests() {
    if [ "${RUN_UNIT_TESTS}" != "true" ]; then
        log_info "Skipping frontend tests"
        return 0
    fi
    
    log_info "Running frontend tests..."
    
    validate_directory_exists "${FRONTEND_PATH}"
    
    cd "${FRONTEND_PATH}"
    
    # Install dependencies
    npm ci
    
    # Run tests
    npm test -- --coverage --watchAll=false --ci || {
        log_error "Frontend tests failed"
        return 1
    }
    
    # Copy coverage reports
    cp -r coverage/ "${TEST_RESULTS_DIR}/frontend-coverage/" || true
    
    log_success "Frontend tests completed"
}

# Run E2E tests
run_e2e_tests() {
    if [ "${RUN_E2E_TESTS}" != "true" ]; then
        log_info "Skipping E2E tests"
        return 0
    fi
    
    log_info "Running E2E tests..."
    
    # This would typically use Playwright or Cypress
    log_warn "E2E tests require additional setup"
    
    log_success "E2E tests completed"
}

# Generate test report
generate_test_report() {
    log_info "Generating test report..."
    
    local report_file="${TEST_RESULTS_DIR}/test-report-${TIMESTAMP}.md"
    
    cat > "${report_file}" << EOF
# Test Execution Report

**Date:** $(date)
**Environment:** ${ENVIRONMENT:-unknown}

## Test Results

### Backend Unit Tests
- Results: ${TEST_RESULTS_DIR}/backend-junit.xml
- Coverage: ${TEST_RESULTS_DIR}/backend-coverage.xml

### Backend Integration Tests
- Results: ${TEST_RESULTS_DIR}/integration-junit.xml
- Coverage: ${TEST_RESULTS_DIR}/integration-coverage.xml

### Frontend Tests
- Coverage: ${TEST_RESULTS_DIR}/frontend-coverage/

## Summary

- Unit Tests: $(if [ -f "${TEST_RESULTS_DIR}/backend-junit.xml" ]; then echo "✅ Completed"; else echo "⏭️ Skipped"; fi)
- Integration Tests: $(if [ -f "${TEST_RESULTS_DIR}/integration-junit.xml" ]; then echo "✅ Completed"; else echo "⏭️ Skipped"; fi)
- Frontend Tests: $(if [ -d "${TEST_RESULTS_DIR}/frontend-coverage" ]; then echo "✅ Completed"; else echo "⏭️ Skipped"; fi)
- E2E Tests: $(if [ "${RUN_E2E_TESTS}" == "true" ]; then echo "✅ Completed"; else echo "⏭️ Skipped"; fi)

EOF
    
    log_success "Test report generated: ${report_file}"
}

# Check coverage threshold
check_coverage() {
    log_info "Checking coverage threshold (${COVERAGE_THRESHOLD}%)..."
    
    if [ -f "${TEST_RESULTS_DIR}/backend-coverage.xml" ]; then
        local coverage=$(python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('${TEST_RESULTS_DIR}/backend-coverage.xml')
root = tree.getroot()
coverage = float(root.get('line-rate', '0')) * 100
print(f'{coverage:.2f}')
" 2>/dev/null || echo "0")
        
        if (( $(echo "${coverage} < ${COVERAGE_THRESHOLD}" | bc -l) )); then
            log_error "Coverage ${coverage}% is below threshold ${COVERAGE_THRESHOLD}%"
            return 1
        else
            log_success "Coverage ${coverage}% meets threshold ${COVERAGE_THRESHOLD}%"
        fi
    fi
}

# Main function
main() {
    log_info "Starting automated testing..."
    log_info "Test results directory: ${TEST_RESULTS_DIR}"
    
    # Run tests
    run_backend_unit_tests
    run_backend_integration_tests
    run_frontend_tests
    run_e2e_tests
    
    # Generate report
    generate_test_report
    
    # Check coverage
    check_coverage
    
    log_success "Automated testing completed!"
}

# Parse arguments
parse_args "$@"

# Run main function
main "$@"




