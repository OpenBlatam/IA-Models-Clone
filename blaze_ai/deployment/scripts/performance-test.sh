#!/bin/bash

# Performance Testing Script for Blaze AI Production
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="http://localhost"
API_PORT="8000"
GRADIO_PORT="8001"
METRICS_PORT="8002"
TEST_DURATION=300
CONCURRENT_USERS=10
RAMP_UP_TIME=60
THINK_TIME=1

# Test data
TEST_PROMPTS=(
    "Generate a creative story about a robot learning to paint"
    "Explain quantum computing in simple terms"
    "Write a poem about artificial intelligence"
    "Create a business plan for a sustainable tech startup"
    "Describe the future of renewable energy"
)

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

check_prerequisites() {
    log_info "Checking performance testing prerequisites..."
    
    # Check if required tools are installed
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi
    
    if ! command -v ab &> /dev/null; then
        log_warn "Apache Bench (ab) not installed, installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y apache2-utils
        elif command -v yum &> /dev/null; then
            sudo yum install -y httpd-tools
        elif command -v brew &> /dev/null; then
            brew install httpd
        else
            log_error "Cannot install Apache Bench automatically"
            exit 1
        fi
    fi
    
    if ! command -v wrk &> /dev/null; then
        log_warn "wrk not installed, installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y wrk
        elif command -v yum &> /dev/null; then
            sudo yum install -y wrk
        elif command -v brew &> /dev/null; then
            brew install wrk
        else
            log_error "Cannot install wrk automatically"
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed"
}

check_service_availability() {
    log_info "Checking service availability..."
    
    # Check API service
    if curl -s -f "$BASE_URL:$API_PORT/health" > /dev/null 2>&1; then
        log_info "API service: AVAILABLE"
    else
        log_error "API service: NOT AVAILABLE"
        exit 1
    fi
    
    # Check Gradio service
    if curl -s -f "$BASE_URL:$GRADIO_PORT/health" > /dev/null 2>&1; then
        log_info "Gradio service: AVAILABLE"
    else
        log_warn "Gradio service: NOT AVAILABLE"
    fi
    
    # Check metrics service
    if curl -s -f "$BASE_URL:$METRICS_PORT/metrics" > /dev/null 2>&1; then
        log_info "Metrics service: AVAILABLE"
    else
        log_warn "Metrics service: NOT AVAILABLE"
    fi
}

test_api_performance() {
    log_info "Testing API performance..."
    
    # Test health endpoint
    log_info "Testing health endpoint performance..."
    ab -n 1000 -c 10 -t 60 "$BASE_URL:$API_PORT/health" > api_health_test.txt 2>&1
    
    # Extract results
    REQUESTS_PER_SEC=$(grep "Requests per second" api_health_test.txt | awk '{print $4}')
    MEAN_TIME=$(grep "Time per request" api_health_test.txt | head -1 | awk '{print $4}')
    FAILED_REQUESTS=$(grep "Failed requests" api_health_test.txt | awk '{print $3}')
    
    log_info "API Health Endpoint Results:"
    log_info "- Requests per second: $REQUESTS_PER_SEC"
    log_info "- Mean time per request: ${MEAN_TIME}ms"
    log_info "- Failed requests: $FAILED_REQUESTS"
    
    # Test with different payloads
    for prompt in "${TEST_PROMPTS[@]}"; do
        log_info "Testing API with prompt: ${prompt:0:50}..."
        
        # Create test payload
        cat > test_payload.json << EOF
{
    "prompt": "$prompt",
    "max_length": 100,
    "temperature": 0.7
}
EOF
        
        # Test with payload
        ab -n 100 -c 5 -p test_payload.json -T application/json -t 30 "$BASE_URL:$API_PORT/generate" > api_generate_test.txt 2>&1
        
        # Extract results
        GEN_REQUESTS_PER_SEC=$(grep "Requests per second" api_generate_test.txt | awk '{print $4}')
        GEN_MEAN_TIME=$(grep "Time per request" api_generate_test.txt | head -1 | awk '{print $4}')
        
        log_info "- Generate endpoint: ${GEN_REQUESTS_PER_SEC} req/s, ${GEN_MEAN_TIME}ms"
    done
}

test_gradio_performance() {
    log_info "Testing Gradio interface performance..."
    
    # Test Gradio health endpoint
    ab -n 500 -c 5 -t 30 "$BASE_URL:$GRADIO_PORT/health" > gradio_health_test.txt 2>&1
    
    # Extract results
    GRADIO_REQUESTS_PER_SEC=$(grep "Requests per second" gradio_health_test.txt | awk '{print $4}')
    GRADIO_MEAN_TIME=$(grep "Time per request" gradio_health_test.txt | head -1 | awk '{print $4}')
    
    log_info "Gradio Health Endpoint Results:"
    log_info "- Requests per second: $GRADIO_REQUESTS_PER_SEC"
    log_info "- Mean time per request: ${GRADIO_MEAN_TIME}ms"
}

test_load_scenarios() {
    log_info "Testing different load scenarios..."
    
    # Light load (10 users, 1 minute)
    log_info "Testing light load (10 users, 1 minute)..."
    wrk -t2 -c10 -d60s -s load_test_script.lua "$BASE_URL:$API_PORT/health" > light_load_test.txt 2>&1
    
    # Medium load (50 users, 2 minutes)
    log_info "Testing medium load (50 users, 2 minutes)..."
    wrk -t4 -c50 -d120s -s load_test_script.lua "$BASE_URL:$API_PORT/health" > medium_load_test.txt 2>&1
    
    # Heavy load (100 users, 3 minutes)
    log_info "Testing heavy load (100 users, 3 minutes)..."
    wrk -t8 -c100 -d180s -s load_test_script.lua "$BASE_URL:$API_PORT/health" > heavy_load_test.txt 2>&1
    
    # Extract results
    log_info "Load Test Results:"
    
    # Light load
    LIGHT_LATENCY=$(grep "Latency" light_load_test.txt | awk '{print $2}')
    LIGHT_REQ_SEC=$(grep "Requests/sec" light_load_test.txt | awk '{print $2}')
    log_info "- Light load: ${LIGHT_LATENCY}ms latency, ${LIGHT_REQ_SEC} req/s"
    
    # Medium load
    MEDIUM_LATENCY=$(grep "Latency" medium_load_test.txt | awk '{print $2}')
    MEDIUM_REQ_SEC=$(grep "Requests/sec" medium_load_test.txt | awk '{print $2}')
    log_info "- Medium load: ${MEDIUM_LATENCY}ms latency, ${MEDIUM_REQ_SEC} req/s"
    
    # Heavy load
    HEAVY_LATENCY=$(grep "Latency" heavy_load_test.txt | awk '{print $2}')
    HEAVY_REQ_SEC=$(grep "Requests/sec" heavy_load_test.txt | awk '{print $2}')
    log_info "- Heavy load: ${HEAVY_LATENCY}ms latency, ${HEAVY_REQ_SEC} req/s"
}

test_stress_scenarios() {
    log_info "Testing stress scenarios..."
    
    # Burst traffic (1000 requests in 10 seconds)
    log_info "Testing burst traffic (1000 requests in 10 seconds)..."
    ab -n 1000 -c 100 -t 10 "$BASE_URL:$API_PORT/health" > burst_test.txt 2>&1
    
    BURST_REQUESTS_PER_SEC=$(grep "Requests per second" burst_test.txt | awk '{print $4}')
    BURST_FAILED=$(grep "Failed requests" burst_test.txt | awk '{print $3}')
    
    log_info "Burst Test Results:"
    log_info "- Requests per second: $BURST_REQUESTS_PER_SEC"
    log_info "- Failed requests: $BURST_FAILED"
    
    # Sustained high load (200 users, 5 minutes)
    log_info "Testing sustained high load (200 users, 5 minutes)..."
    wrk -t10 -c200 -d300s -s load_test_script.lua "$BASE_URL:$API_PORT/health" > sustained_load_test.txt 2>&1
    
    SUSTAINED_LATENCY=$(grep "Latency" sustained_load_test.txt | awk '{print $2}')
    SUSTAINED_REQ_SEC=$(grep "Requests/sec" sustained_load_test.txt | awk '{print $2}')
    
    log_info "Sustained Load Test Results:"
    log_info "- Latency: ${SUSTAINED_LATENCY}ms"
    log_info "- Requests per second: ${SUSTAINED_REQ_SEC}"
}

test_memory_and_cpu() {
    log_info "Testing memory and CPU usage under load..."
    
    # Start background monitoring
    log_info "Starting resource monitoring..."
    
    # Monitor system resources during test
    (
        while true; do
            echo "$(date '+%H:%M:%S'),$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1),$(free | grep Mem | awk '{print $3/$2 * 100.0}')" >> resource_usage.csv
            sleep 5
        done
    ) &
    MONITOR_PID=$!
    
    # Run load test
    log_info "Running load test while monitoring resources..."
    wrk -t4 -c50 -d120s -s load_test_script.lua "$BASE_URL:$API_PORT/health" > resource_test.txt 2>&1
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    
    # Analyze resource usage
    log_info "Resource Usage Analysis:"
    
    if [ -f "resource_usage.csv" ]; then
        MAX_CPU=$(awk -F',' 'NR>1 {print $2}' resource_usage.csv | sort -n | tail -1)
        MAX_MEMORY=$(awk -F',' 'NR>1 {print $3}' resource_usage.csv | sort -n | tail -1)
        AVG_CPU=$(awk -F',' 'NR>1 {sum+=$2} END {print sum/(NR-1)}' resource_usage.csv)
        AVG_MEMORY=$(awk -F',' 'NR>1 {sum+=$3} END {print sum/(NR-1)}' resource_usage.csv)
        
        log_info "- Peak CPU usage: ${MAX_CPU}%"
        log_info "- Peak memory usage: ${MAX_MEMORY}%"
        log_info "- Average CPU usage: ${AVG_CPU}%"
        log_info "- Average memory usage: ${AVG_MEMORY}%"
    fi
}

test_database_performance() {
    log_info "Testing database performance..."
    
    # Test database connection pool
    if command -v psql &> /dev/null && [ -n "$DB_PASSWORD" ]; then
        log_info "Testing PostgreSQL connection pool..."
        
        # Test multiple concurrent connections
        for i in {1..10}; do
            PGPASSWORD=$DB_PASSWORD psql -h localhost -U blazeai -d blazeai -c "SELECT 1;" > /dev/null 2>&1 &
        done
        wait
        
        log_info "PostgreSQL connection pool test completed"
    else
        log_warn "PostgreSQL not available for testing"
    fi
    
    # Test Redis performance
    if command -v redis-cli &> /dev/null && [ -n "$REDIS_PASSWORD" ]; then
        log_info "Testing Redis performance..."
        
        # Test Redis ping
        redis-cli -a $REDIS_PASSWORD ping > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            log_info "Redis ping: OK"
        else
            log_warn "Redis ping: FAILED"
        fi
        
        # Test Redis set/get operations
        redis-cli -a $REDIS_PASSWORD set "test_key" "test_value" > /dev/null 2>&1
        redis-cli -a $REDIS_PASSWORD get "test_key" > /dev/null 2>&1
        redis-cli -a $REDIS_PASSWORD del "test_key" > /dev/null 2>&1
        
        log_info "Redis operations test completed"
    else
        log_warn "Redis not available for testing"
    fi
}

create_load_test_script() {
    log_info "Creating load test script..."
    
    cat > load_test_script.lua << 'EOF'
-- Load test script for Blaze AI
wrk.method = "GET"
wrk.headers = {
    ["Content-Type"] = "application/json",
    ["User-Agent"] = "BlazeAI-PerformanceTest/1.0"
}

function request()
    return wrk.format("GET", "/health")
end

function response(status, headers, body)
    if status ~= 200 then
        print("Request failed with status: " .. status)
    end
end
EOF
    
    log_info "Load test script created"
}

generate_performance_report() {
    log_info "Generating performance test report..."
    
    REPORT_FILE="performance_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > $REPORT_FILE << EOF
Blaze AI Performance Test Report
================================
Generated: $(date)
Test Duration: ${TEST_DURATION} seconds
Concurrent Users: ${CONCURRENT_USERS}
Ramp-up Time: ${RAMP_UP_TIME} seconds

API Performance Results:
- Health Endpoint: ${REQUESTS_PER_SEC} req/s, ${MEAN_TIME}ms
- Failed Requests: ${FAILED_REQUESTS}

Gradio Performance Results:
- Health Endpoint: ${GRADIO_REQUESTS_PER_SEC} req/s, ${GRADIO_MEAN_TIME}ms

Load Test Results:
- Light Load (10 users): ${LIGHT_LATENCY}ms latency, ${LIGHT_REQ_SEC} req/s
- Medium Load (50 users): ${MEDIUM_LATENCY}ms latency, ${MEDIUM_REQ_SEC} req/s
- Heavy Load (100 users): ${HEAVY_LATENCY}ms latency, ${HEAVY_REQ_SEC} req/s

Stress Test Results:
- Burst Traffic: ${BURST_REQUESTS_PER_SEC} req/s, ${BURST_FAILED} failed
- Sustained Load: ${SUSTAINED_LATENCY}ms latency, ${SUSTAINED_REQ_SEC} req/s

Resource Usage:
- Peak CPU: ${MAX_CPU}%
- Peak Memory: ${MAX_MEMORY}%
- Average CPU: ${AVG_CPU}%
- Average Memory: ${AVG_MEMORY}%

Recommendations:
$(if (( $(echo "$REQUESTS_PER_SEC < 100" | bc -l) )); then echo "- Consider optimizing API performance"; fi)
$(if (( $(echo "$MEAN_TIME > 1000" | bc -l) )); then echo "- High response times detected, investigate bottlenecks"; fi)
$(if [ "$FAILED_REQUESTS" -gt 0 ]; then echo "- Failed requests detected, check error logs"; fi)
$(if (( $(echo "$MAX_CPU > 80" | bc -l) )); then echo "- High CPU usage, consider scaling up"; fi)
$(if (( $(echo "$MAX_MEMORY > 80" | bc -l) )); then echo "- High memory usage, consider scaling up"; fi)
EOF
    
    log_info "Performance report generated: $REPORT_FILE"
}

cleanup() {
    log_info "Cleaning up test files..."
    
    # Remove test files
    rm -f test_payload.json
    rm -f api_health_test.txt
    rm -f api_generate_test.txt
    rm -f gradio_health_test.txt
    rm -f light_load_test.txt
    rm -f medium_load_test.txt
    rm -f heavy_load_test.txt
    rm -f burst_test.txt
    rm -f sustained_load_test.txt
    rm -f resource_test.txt
    rm -f resource_usage.csv
    rm -f load_test_script.lua
    
    log_info "Cleanup completed"
}

# Main performance testing logic
main() {
    log_info "Starting Blaze AI performance testing..."
    
    check_prerequisites
    check_service_availability
    create_load_test_script
    
    # Run performance tests
    test_api_performance
    test_gradio_performance
    test_load_scenarios
    test_stress_scenarios
    test_memory_and_cpu
    test_database_performance
    
    # Generate report
    generate_performance_report
    
    # Cleanup
    cleanup
    
    log_info "Performance testing completed successfully!"
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
