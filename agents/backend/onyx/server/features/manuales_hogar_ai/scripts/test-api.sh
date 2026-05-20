#!/bin/bash
# API Test script - Test all API endpoints
# Usage: ./scripts/test-api.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"
PASSED=0
FAILED=0

echo -e "${BLUE}🧪 Testing Manuales Hogar AI API${NC}"
echo "API URL: $API_URL"
echo ""

# Test function
test_endpoint() {
    local method=$1
    local endpoint=$2
    local description=$3
    local data=$4
    
    echo -n "Testing $description... "
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint" 2>/dev/null)
    elif [ "$method" == "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$API_URL$endpoint" 2>/dev/null)
    fi
    
    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        echo -e "${GREEN}✅ PASS${NC} (HTTP $http_code)"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $http_code)"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Test endpoints
test_endpoint "GET" "/" "Root endpoint"
test_endpoint "GET" "/api/v1/health" "Health check"
test_endpoint "GET" "/api/v1/models" "List models"
test_endpoint "GET" "/api/v1/categories" "List categories"

# Test with authentication if API key is set
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo ""
    echo -e "${BLUE}Testing endpoints that require API key...${NC}"
    
    # Note: These may fail if API key is invalid, but we test the endpoint structure
    test_endpoint "POST" "/api/v1/generate-from-text" "Generate from text" \
        '{"problem_description": "Test problem", "category": "general"}'
else
    echo ""
    echo -e "${YELLOW}⚠️  OPENROUTER_API_KEY not set. Skipping authenticated endpoints.${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}=== Test Summary ===${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi




