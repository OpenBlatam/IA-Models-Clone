#!/bin/bash
# Quick test script - Fast API validation
# Usage: ./scripts/quick-test.sh

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"

echo -e "${BLUE}⚡ Quick API Test${NC}"
echo ""

# Quick health check
if curl -f "$API_URL/api/v1/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Service is running${NC}"
    
    # Get response time
    RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "$API_URL/api/v1/health")
    echo "Response time: ${RESPONSE_TIME}s"
    
    # Test root
    if curl -f "$API_URL/" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Root endpoint OK${NC}"
    else
        echo -e "${RED}❌ Root endpoint failed${NC}"
    fi
    
    # Test docs
    if curl -f "$API_URL/docs" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API docs accessible${NC}"
    else
        echo -e "${YELLOW}⚠️  API docs not accessible${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Quick test passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Service is not responding${NC}"
    echo ""
    echo "Check if services are running:"
    echo "  docker-compose ps"
    echo ""
    echo "Start services:"
    echo "  ./start.sh"
    exit 1
fi




