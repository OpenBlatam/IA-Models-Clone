#!/bin/bash

set -e

echo "🚀 Building GitHub Autonomous Agent Go Services"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go 1.22 or later."
    exit 1
fi

echo -e "${BLUE}📦 Downloading dependencies...${NC}"
go mod download
go mod tidy

echo -e "${BLUE}🔨 Building binary...${NC}"
go build -o agent-service ./cmd/agent

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo -e "${GREEN}   Binary: ./agent-service${NC}"
    echo ""
    echo "To run: ./agent-service --port 8080"
else
    echo "❌ Build failed!"
    exit 1
fi












