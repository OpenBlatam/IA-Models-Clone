#!/bin/bash
# Build script for Transcriber Core

set -e

echo "🦀 Building Transcriber Core..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "📦 Installing maturin..."
    pip install maturin
fi

# Build mode
MODE=${1:-release}

if [ "$MODE" = "release" ]; then
    echo -e "${BLUE}Building in RELEASE mode...${NC}"
    maturin build --release
    echo -e "${GREEN}✓ Release build complete!${NC}"
elif [ "$MODE" = "dev" ]; then
    echo -e "${BLUE}Building in DEV mode...${NC}"
    maturin develop
    echo -e "${GREEN}✓ Dev build complete!${NC}"
else
    echo "Usage: ./build.sh [release|dev]"
    exit 1
fi

# Run tests
echo -e "${BLUE}Running tests...${NC}"
cargo test --lib

echo -e "${GREEN}✓ All done!${NC}"












