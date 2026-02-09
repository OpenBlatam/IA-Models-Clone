#!/bin/bash
# Automated Performance Testing Script
# Runs load tests, stress tests, and generates reports

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_URL="${TEST_URL:-http://localhost:8000}"
readonly RESULTS_DIR="${RESULTS_DIR:-./performance-results}"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Test parameters
readonly CONCURRENT_USERS="${CONCURRENT_USERS:-50}"
readonly DURATION="${DURATION:-300}"
readonly RAMP_UP_TIME="${RAMP_UP_TIME:-60}"
readonly MAX_RESPONSE_TIME="${MAX_RESPONSE_TIME:-2000}"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Install k6 if not present
install_k6() {
    if command -v k6 &> /dev/null; then
        return 0
    fi
    
    log_info "Installing k6..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo gpg -k
        sudo gpg --no-default-keyring \
            --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
            --keyserver hkp://keyserver.ubuntu.com:80 \
            --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D9B
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
            sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install k6
    fi
}

# Create k6 test script
create_k6_script() {
    local script_file="${RESULTS_DIR}/k6-test-${TIMESTAMP}.js"
    
    cat > "${script_file}" << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

export const options = {
  stages: [
    { duration: '60s', target: 10 },   // Ramp up
    { duration: '240s', target: 50 },  // Stay at 50 users
    { duration: '60s', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
  },
};

export default function () {
  const baseUrl = __ENV.TEST_URL || 'http://localhost:8000';
  
  // Test health endpoint
  const healthRes = http.get(`${baseUrl}/health`);
  check(healthRes, {
    'health status is 200': (r) => r.status === 200,
    'health response time < 500ms': (r) => r.timings.duration < 500,
  });
  errorRate.add(healthRes.status !== 200);
  responseTime.add(healthRes.timings.duration);
  
  sleep(1);
  
  // Test API endpoint
  const apiRes = http.get(`${baseUrl}/api/music/search?q=test`);
  check(apiRes, {
    'api status is 200': (r) => r.status === 200,
    'api response time < 2000ms': (r) => r.timings.duration < 2000,
  });
  errorRate.add(apiRes.status !== 200);
  responseTime.add(apiRes.timings.duration);
  
  sleep(1);
}
EOF
    
    echo "${script_file}"
}

# Run load test
run_load_test() {
    log_info "Starting load test..."
    log_info "URL: ${TEST_URL}"
    log_info "Concurrent users: ${CONCURRENT_USERS}"
    log_info "Duration: ${DURATION}s"
    
    mkdir -p "${RESULTS_DIR}"
    local script_file=$(create_k6_script)
    
    install_k6
    
    log_info "Running k6 load test..."
    k6 run \
        --vus "${CONCURRENT_USERS}" \
        --duration "${DURATION}s" \
        --env TEST_URL="${TEST_URL}" \
        --out json="${RESULTS_DIR}/k6-results-${TIMESTAMP}.json" \
        --out html="${RESULTS_DIR}/k6-report-${TIMESTAMP}.html" \
        "${script_file}" || {
        log_error "Load test failed"
        return 1
    }
    
    log_success "Load test completed"
    log_info "Results saved to: ${RESULTS_DIR}/k6-report-${TIMESTAMP}.html"
}

# Run stress test
run_stress_test() {
    log_info "Starting stress test..."
    
    mkdir -p "${RESULTS_DIR}"
    local script_file=$(create_k6_script)
    
    install_k6
    
    log_info "Running k6 stress test (finding breaking point)..."
    k6 run \
        --vus 1 \
        --iterations 1 \
        --env TEST_URL="${TEST_URL}" \
        "${script_file}"
    
    # Gradually increase load until failure
    local vus=10
    while [ ${vus} -le 1000 ]; do
        log_info "Testing with ${vus} virtual users..."
        
        if ! k6 run \
            --vus ${vus} \
            --duration 30s \
            --env TEST_URL="${TEST_URL}" \
            "${script_file}" 2>&1 | grep -q "checks.*100.00%"; then
            log_warn "Breaking point found at ${vus} virtual users"
            break
        fi
        
        vus=$((vus + 50))
    done
    
    log_success "Stress test completed"
}

# Run spike test
run_spike_test() {
    log_info "Starting spike test..."
    
    mkdir -p "${RESULTS_DIR}"
    local script_file=$(create_k6_script)
    
    install_k6
    
    log_info "Running spike test (sudden traffic increase)..."
    k6 run \
        --vus 200 \
        --duration 60s \
        --env TEST_URL="${TEST_URL}" \
        --out json="${RESULTS_DIR}/k6-spike-${TIMESTAMP}.json" \
        "${script_file}" || true
    
    log_success "Spike test completed"
}

# Analyze results
analyze_results() {
    log_info "Analyzing performance test results..."
    
    if [ ! -f "${RESULTS_DIR}/k6-results-${TIMESTAMP}.json" ]; then
        log_error "No results file found"
        return 1
    fi
    
    # Extract key metrics using jq
    if command -v jq &> /dev/null; then
        local avg_response_time=$(jq -r '.metrics.http_req_duration.values.avg' \
            "${RESULTS_DIR}/k6-results-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
        local p95_response_time=$(jq -r '.metrics.http_req_duration.values["p(95)"]' \
            "${RESULTS_DIR}/k6-results-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
        local error_rate=$(jq -r '.metrics.http_req_failed.values.rate' \
            "${RESULTS_DIR}/k6-results-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
        
        log_info "=== Performance Test Results ==="
        log_info "Average Response Time: ${avg_response_time}ms"
        log_info "95th Percentile: ${p95_response_time}ms"
        log_info "Error Rate: $(echo "${error_rate} * 100" | bc -l)%"
        
        # Check thresholds
        if (( $(echo "${p95_response_time} > ${MAX_RESPONSE_TIME}" | bc -l) )); then
            log_error "95th percentile response time exceeds threshold: ${p95_response_time}ms > ${MAX_RESPONSE_TIME}ms"
            return 1
        fi
    fi
    
    log_success "Performance analysis completed"
}

# Generate report
generate_report() {
    log_info "Generating performance test report..."
    
    local report_file="${RESULTS_DIR}/performance-report-${TIMESTAMP}.md"
    
    cat > "${report_file}" << EOF
# Performance Test Report

**Date:** $(date)
**Test URL:** ${TEST_URL}
**Concurrent Users:** ${CONCURRENT_USERS}
**Duration:** ${DURATION}s

## Results

- HTML Report: k6-report-${TIMESTAMP}.html
- JSON Results: k6-results-${TIMESTAMP}.json

## Recommendations

1. Review response times and optimize slow endpoints
2. Check error rates and fix failing requests
3. Monitor resource usage during tests
4. Consider scaling if thresholds are exceeded

EOF
    
    log_success "Report generated: ${report_file}"
}

# Main function
main() {
    case "${1:-load}" in
        load)
            run_load_test
            analyze_results
            generate_report
            ;;
        stress)
            run_stress_test
            ;;
        spike)
            run_spike_test
            ;;
        analyze)
            analyze_results
            ;;
        *)
            echo "Usage: $0 {load|stress|spike|analyze}"
            echo ""
            echo "Commands:"
            echo "  load    - Run load test (default)"
            echo "  stress  - Run stress test to find breaking point"
            echo "  spike   - Run spike test with sudden traffic increase"
            echo "  analyze - Analyze existing test results"
            exit 1
            ;;
    esac
}

main "$@"




