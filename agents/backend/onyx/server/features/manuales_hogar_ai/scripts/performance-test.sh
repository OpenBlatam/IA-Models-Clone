#!/bin/bash
# Performance test script - Load testing
# Usage: ./scripts/performance-test.sh [--concurrent N] [--requests N]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"
CONCURRENT="${CONCURRENT:-10}"
REQUESTS="${REQUESTS:-100}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        --requests)
            REQUESTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}⚡ Performance Test - Manuales Hogar AI${NC}"
echo "API URL: $API_URL"
echo "Concurrent requests: $CONCURRENT"
echo "Total requests: $REQUESTS"
echo ""

# Check if ab (Apache Bench) is available
if ! command -v ab &> /dev/null; then
    echo -e "${YELLOW}⚠️  Apache Bench (ab) not found. Using curl-based test...${NC}"
    
    # Simple curl-based test
    echo "Running simple performance test..."
    START_TIME=$(date +%s.%N)
    
    SUCCESS=0
    FAILED=0
    TOTAL_TIME=0
    
    for i in $(seq 1 $REQUESTS); do
        REQUEST_START=$(date +%s.%N)
        if curl -f -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
            REQUEST_END=$(date +%s.%N)
            REQUEST_TIME=$(echo "$REQUEST_END - $REQUEST_START" | bc)
            TOTAL_TIME=$(echo "$TOTAL_TIME + $REQUEST_TIME" | bc)
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        
        if [ $((i % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    AVG_TIME=$(echo "scale=3; $TOTAL_TIME / $SUCCESS" | bc)
    RPS=$(echo "scale=2; $SUCCESS / $ELAPSED" | bc)
    
    echo ""
    echo ""
    echo -e "${BLUE}=== Results ===${NC}"
    echo "Total time: ${ELAPSED}s"
    echo "Successful: $SUCCESS"
    echo "Failed: $FAILED"
    echo "Average response time: ${AVG_TIME}s"
    echo "Requests per second: $RPS"
    
else
    # Use Apache Bench
    echo "Running Apache Bench test..."
    ab -n $REQUESTS -c $CONCURRENT "$API_URL/api/v1/health" | tee /tmp/ab_results.txt
    
    echo ""
    echo -e "${BLUE}=== Summary ===${NC}"
    grep "Requests per second" /tmp/ab_results.txt || true
    grep "Time per request" /tmp/ab_results.txt || true
    grep "Failed requests" /tmp/ab_results.txt || true
fi

echo ""




