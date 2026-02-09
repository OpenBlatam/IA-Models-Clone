#!/bin/bash
# Performance testing script
# Tests application performance under load

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Default values
readonly TARGET_URL="${TARGET_URL:-}"
readonly CONCURRENT_USERS="${CONCURRENT_USERS:-10}"
readonly DURATION="${DURATION:-60}"
readonly RAMP_UP="${RAMP_UP:-10}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-./performance_results}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run performance tests against the application.

OPTIONS:
    -u, --url URL            Target URL (required)
    -c, --concurrent NUM     Concurrent users (default: 10)
    -d, --duration SECONDS    Test duration in seconds (default: 60)
    -r, --ramp-up SECONDS    Ramp-up time in seconds (default: 10)
    -o, --output DIR         Output directory (default: ./performance_results)
    -h, --help               Show this help message

EXAMPLES:
    $0 --url http://1.2.3.4:8030 --concurrent 50 --duration 300
    $0 --url http://1.2.3.4:8030/api/health --concurrent 100

EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--url)
                TARGET_URL="$2"
                shift 2
                ;;
            -c|--concurrent)
                CONCURRENT_USERS="$2"
                shift 2
                ;;
            -d|--duration)
                DURATION="$2"
                shift 2
                ;;
            -r|--ramp-up)
                RAMP_UP="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
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

# Check if Apache Bench is available
check_ab() {
    if ! command -v ab &> /dev/null; then
        log_warn "Apache Bench (ab) not found. Installing..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y apache2-utils
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install apache-bench
        else
            error_exit 1 "Apache Bench not available. Please install manually."
        fi
    fi
}

# Run Apache Bench test
run_ab_test() {
    local url="${1}"
    local concurrent="${2}"
    local requests="${3}"
    
    log_info "Running Apache Bench test..."
    log_info "URL: ${url}"
    log_info "Concurrent users: ${concurrent}"
    log_info "Total requests: ${requests}"
    
    mkdir -p "${OUTPUT_DIR}"
    local output_file="${OUTPUT_DIR}/ab_results_$(date +%Y%m%d_%H%M%S).txt"
    
    ab -n "${requests}" \
       -c "${concurrent}" \
       -g "${OUTPUT_DIR}/ab_plot_data.tsv" \
       -e "${OUTPUT_DIR}/ab_csv_data.csv" \
       -k \
       -v 2 \
       "${url}" > "${output_file}" 2>&1
    
    echo "${output_file}"
}

# Run curl-based test
run_curl_test() {
    local url="${1}"
    local concurrent="${2}"
    local duration="${3}"
    
    log_info "Running curl-based performance test..."
    
    mkdir -p "${OUTPUT_DIR}"
    local results_file="${OUTPUT_DIR}/curl_results_$(date +%Y%m%d_%H%M%S).txt"
    
    local end_time
    end_time=$(($(date +%s) + duration))
    local request_count=0
    local success_count=0
    local fail_count=0
    local total_time=0
    local min_time=999999
    local max_time=0
    
    log_info "Test running for ${duration} seconds..."
    
    while [ $(date +%s) -lt $end_time ]; do
        local start
        start=$(date +%s.%N)
        
        if curl -sf -m 10 "${url}" > /dev/null 2>&1; then
            local end
            end=$(date +%s.%N)
            local elapsed
            elapsed=$(echo "$end - $start" | bc)
            local elapsed_ms
            elapsed_ms=$(echo "$elapsed * 1000" | bc | awk '{printf "%.2f", $1}')
            
            success_count=$((success_count + 1))
            total_time=$(echo "$total_time + $elapsed_ms" | bc)
            
            local elapsed_float
            elapsed_float=$(echo "$elapsed_ms" | awk '{print int($1)}')
            if (( $(echo "$elapsed_float < $min_time" | bc -l) )); then
                min_time=$elapsed_float
            fi
            if (( $(echo "$elapsed_float > $max_time" | bc -l) )); then
                max_time=$elapsed_float
            fi
        else
            fail_count=$((fail_count + 1))
        fi
        
        request_count=$((request_count + 1))
        
        # Limit concurrent requests
        if [ $((request_count % concurrent)) -eq 0 ]; then
            sleep 0.1
        fi
    done
    
    local avg_time=0
    if [ $success_count -gt 0 ]; then
        avg_time=$(echo "scale=2; $total_time / $success_count" | bc)
    fi
    
    local success_rate=0
    if [ $request_count -gt 0 ]; then
        success_rate=$(echo "scale=2; ($success_count * 100) / $request_count" | bc)
    fi
    
    cat > "${results_file}" << EOF
Performance Test Results
=======================
URL: ${url}
Duration: ${duration} seconds
Concurrent Users: ${concurrent}

Total Requests: ${request_count}
Successful: ${success_count}
Failed: ${fail_count}
Success Rate: ${success_rate}%

Response Times:
  Min: ${min_time}ms
  Max: ${max_time}ms
  Avg: ${avg_time}ms

Requests per second: $(echo "scale=2; $request_count / $duration" | bc)
EOF
    
    echo "${results_file}"
}

# Display results
display_results() {
    local results_file="${1}"
    
    echo ""
    echo "${GREEN}==========================================${NC}"
    echo "${GREEN}Performance Test Results${NC}"
    echo "${GREEN}==========================================${NC}"
    echo ""
    cat "${results_file}"
    echo ""
    echo "${GREEN}==========================================${NC}"
    echo "Results saved to: ${results_file}"
    echo "${GREEN}==========================================${NC}"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${TARGET_URL}" ]; then
        error_exit 1 "TARGET_URL is required"
    fi
    
    # Calculate total requests
    local total_requests
    total_requests=$((CONCURRENT_USERS * DURATION / 2))  # Rough estimate
    
    log_info "Starting performance test..."
    log_info "Target: ${TARGET_URL}"
    log_info "Concurrent users: ${CONCURRENT_USERS}"
    log_info "Duration: ${DURATION}s"
    log_info "Ramp-up: ${RAMP_UP}s"
    
    # Try Apache Bench first, fallback to curl
    if command -v ab &> /dev/null; then
        local results_file
        results_file=$(run_ab_test "${TARGET_URL}" "${CONCURRENT_USERS}" "${total_requests}")
        display_results "${results_file}"
    else
        log_info "Apache Bench not available, using curl-based test..."
        local results_file
        results_file=$(run_curl_test "${TARGET_URL}" "${CONCURRENT_USERS}" "${DURATION}")
        display_results "${results_file}"
    fi
}

main "$@"


