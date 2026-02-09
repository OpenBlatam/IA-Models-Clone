#!/bin/bash
# Code quality check script

set -e

echo "🔍 Running code quality checks..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Format check
echo -e "${BLUE}Checking code format...${NC}"
if cargo fmt --check; then
    echo -e "${GREEN}✓ Format OK${NC}"
else
    echo -e "${RED}✗ Format issues found. Run 'cargo fmt'${NC}"
    exit 1
fi

# Clippy check
echo -e "${BLUE}Running clippy...${NC}"
if cargo clippy -- -D warnings; then
    echo -e "${GREEN}✓ Clippy OK${NC}"
else
    echo -e "${RED}✗ Clippy issues found${NC}"
    exit 1
fi

# Tests
echo -e "${BLUE}Running tests...${NC}"
cargo test --lib

echo -e "${GREEN}✓ All checks passed!${NC}"












