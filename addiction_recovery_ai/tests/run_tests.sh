#!/bin/bash
# Test runner script for addiction_recovery_ai

set -e

echo "=========================================="
echo "Running Addiction Recovery AI Tests"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install it with: pip install -r requirements-dev.txt"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-all}"
COVERAGE="${2:-true}"

echo -e "${YELLOW}Test Type: ${TEST_TYPE}${NC}"
echo -e "${YELLOW}Coverage: ${COVERAGE}${NC}"
echo ""

# Change to tests directory
cd "$(dirname "$0")"

# Base pytest command
PYTEST_CMD="pytest"

# Add coverage if requested
if [ "$COVERAGE" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=.. --cov-report=html --cov-report=term-missing"
fi

# Run tests based on type
case "$TEST_TYPE" in
    unit)
        echo -e "${GREEN}Running unit tests...${NC}"
        $PYTEST_CMD -m "not integration and not slow" test_*.py
        ;;
    integration)
        echo -e "${GREEN}Running integration tests...${NC}"
        $PYTEST_CMD -m integration test_integration*.py
        ;;
    api)
        echo -e "${GREEN}Running API tests...${NC}"
        $PYTEST_CMD test_api_endpoints.py
        ;;
    services)
        echo -e "${GREEN}Running service tests...${NC}"
        $PYTEST_CMD test_services.py
        ;;
    middleware)
        echo -e "${GREEN}Running middleware tests...${NC}"
        $PYTEST_CMD test_middleware.py
        ;;
    fast)
        echo -e "${GREEN}Running fast tests...${NC}"
        $PYTEST_CMD -m "not slow" test_*.py
        ;;
    all)
        echo -e "${GREEN}Running all tests...${NC}"
        $PYTEST_CMD test_*.py
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Usage: $0 [unit|integration|api|services|middleware|fast|all] [coverage:true|false]"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo -e "All tests passed!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    
    if [ "$COVERAGE" = "true" ]; then
        echo ""
        echo -e "${YELLOW}Coverage report generated in htmlcov/index.html${NC}"
    fi
else
    echo ""
    echo -e "${RED}=========================================="
    echo -e "Some tests failed!${NC}"
    echo -e "${RED}==========================================${NC}"
    exit 1
fi



