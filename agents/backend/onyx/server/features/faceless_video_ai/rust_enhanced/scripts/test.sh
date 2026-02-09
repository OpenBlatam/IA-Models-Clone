#!/bin/bash

set -e

echo "🧪 Running tests for Rust Enhanced Core"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Running Rust tests...${NC}"
cargo test --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All Rust tests passed!${NC}"
else
    echo -e "${RED}❌ Rust tests failed!${NC}"
    exit 1
fi

echo -e "${BLUE}Running benchmarks...${NC}"
cargo bench

echo -e "${GREEN}✅ Tests and benchmarks completed!${NC}"












