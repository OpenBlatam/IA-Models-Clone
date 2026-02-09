#!/bin/bash
# Test script for Transcriber Core

set -e

echo "🧪 Running Transcriber Core Tests..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

TEST_TYPE=${1:-all}

case $TEST_TYPE in
    unit)
        echo -e "${BLUE}Running unit tests...${NC}"
        cargo test --lib
        ;;
    integration)
        echo -e "${BLUE}Running integration tests...${NC}"
        cargo test --test integration_tests
        ;;
    bench)
        echo -e "${BLUE}Running benchmarks...${NC}"
        cargo bench
        ;;
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        cargo test --lib
        cargo test --test integration_tests
        echo -e "${YELLOW}Note: Run 'cargo bench' separately for benchmarks${NC}"
        ;;
    *)
        echo "Usage: ./test.sh [unit|integration|bench|all]"
        exit 1
        ;;
esac

echo -e "${GREEN}✓ Tests complete!${NC}"












