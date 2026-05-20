#!/bin/bash
# Performance Testing Script
# Runs load tests before deployment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_URL="${1:-http://localhost:8020}"
CONCURRENT_USERS="${2:-10}"
DURATION="${3:-60}"
REQUESTS_PER_SECOND="${4:-100}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check if required tools are available
check_tools() {
    local missing_tools=()
    
    # Check for Apache Bench (ab) or wrk
    if ! command -v ab > /dev/null 2>&1 && ! command -v wrk > /dev/null 2>&1; then
        missing_tools+=("ab or wrk")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install with: sudo apt-get install apache2-utils (for ab) or wrk"
        return 1
    fi
    
    return 0
}

# Run Apache Bench test
run_ab_test() {
    local total_requests=$((REQUESTS_PER_SECOND * DURATION))
    
    log_info "Running Apache Bench test..."
    log_info "URL: $TEST_URL"
    log_info "Concurrent users: $CONCURRENT_USERS"
    log_info "Duration: ${DURATION}s"
    log_info "Total requests: $total_requests"
    
    ab -n "$total_requests" \
       -c "$CONCURRENT_USERS" \
       -t "$DURATION" \
       -k \
       -q \
       "$TEST_URL/health" > /tmp/ab_results.txt 2>&1
    
    if [ $? -eq 0 ]; then
        log_info "✅ Performance test completed"
        cat /tmp/ab_results.txt
        return 0
    else
        log_error "❌ Performance test failed"
        cat /tmp/ab_results.txt
        return 1
    fi
}

# Run wrk test
run_wrk_test() {
    log_info "Running wrk test..."
    log_info "URL: $TEST_URL"
    log_info "Threads: $CONCURRENT_USERS"
    log_info "Connections: $CONCURRENT_USERS"
    log_info "Duration: ${DURATION}s"
    
    wrk -t"$CONCURRENT_USERS" \
        -c"$CONCURRENT_USERS" \
        -d"${DURATION}s" \
        --latency \
        "$TEST_URL/health" > /tmp/wrk_results.txt 2>&1
    
    if [ $? -eq 0 ]; then
        log_info "✅ Performance test completed"
        cat /tmp/wrk_results.txt
        return 0
    else
        log_error "❌ Performance test failed"
        cat /tmp/wrk_results.txt
        return 1
    fi
}

# Analyze results
analyze_results() {
    log_info "Analyzing performance test results..."
    
    local results_file=""
    if [ -f "/tmp/wrk_results.txt" ]; then
        results_file="/tmp/wrk_results.txt"
    elif [ -f "/tmp/ab_results.txt" ]; then
        results_file="/tmp/ab_results.txt"
    else
        log_error "No results file found"
        return 1
    fi
    
    # Extract key metrics
    local avg_latency=$(grep -E "Time per request|Latency" "$results_file" | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0")
    local requests_per_sec=$(grep -E "Requests per second|Requests/sec" "$results_file" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0")
    local failed_requests=$(grep -E "Failed requests|Non-2xx" "$results_file" | grep -oE '[0-9]+' | head -1 || echo "0")
    
    log_info "Performance Metrics:"
    log_info "  Average Latency: ${avg_latency}ms"
    log_info "  Requests/sec: $requests_per_sec"
    log_info "  Failed Requests: $failed_requests"
    
    # Check thresholds
    local latency_threshold=500  # 500ms
    local min_rps=50  # Minimum requests per second
    
    if (( $(echo "$avg_latency > $latency_threshold" | bc -l) )); then
        log_warn "⚠️  High latency detected: ${avg_latency}ms (threshold: ${latency_threshold}ms)"
        return 1
    fi
    
    if (( $(echo "$requests_per_sec < $min_rps" | bc -l) )); then
        log_warn "⚠️  Low throughput: ${requests_per_sec} req/s (minimum: ${min_rps} req/s)"
        return 1
    fi
    
    if [ "$failed_requests" -gt 0 ]; then
        log_warn "⚠️  Failed requests detected: $failed_requests"
        return 1
    fi
    
    log_info "✅ Performance test passed all thresholds"
    return 0
}

# Main function
main() {
    log_info "=== Performance Testing ==="
    log_info "Target: $TEST_URL"
    log_info "Concurrent Users: $CONCURRENT_USERS"
    log_info "Duration: ${DURATION}s"
    
    if ! check_tools; then
        exit 1
    fi
    
    # Pre-test health check
    log_info "Pre-test health check..."
    if ! curl -f -s "$TEST_URL/health" > /dev/null 2>&1; then
        log_error "Target is not healthy"
        exit 1
    fi
    
    # Run test
    if command -v wrk > /dev/null 2>&1; then
        if ! run_wrk_test; then
            exit 1
        fi
    elif command -v ab > /dev/null 2>&1; then
        if ! run_ab_test; then
            exit 1
        fi
    fi
    
    # Analyze results
    if ! analyze_results; then
        log_error "Performance test failed thresholds"
        exit 1
    fi
    
    log_info "=== Performance test completed successfully ==="
}

main "$@"

