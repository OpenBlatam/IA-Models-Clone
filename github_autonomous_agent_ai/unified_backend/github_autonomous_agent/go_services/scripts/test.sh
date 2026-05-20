#!/bin/bash

set -e

echo "🧪 Running tests for Go Services"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Running unit tests...${NC}"
go test -v ./...

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi

echo -e "${BLUE}Running tests with coverage...${NC}"
go test -v -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html

echo -e "${GREEN}✅ Coverage report generated: coverage.html${NC}"












