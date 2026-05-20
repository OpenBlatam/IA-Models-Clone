#!/bin/bash

set -e

echo "🦀 Building Faceless Video AI Rust Enhanced Core"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust is not installed. Please install Rust."
    exit 1
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo -e "${BLUE}📦 Installing maturin...${NC}"
    pip install maturin
fi

echo -e "${BLUE}🔨 Building in release mode...${NC}"
maturin develop --release

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""
    echo "To use in Python:"
    echo "  from faceless_video_enhanced import EffectsEngine"
else
    echo "❌ Build failed!"
    exit 1
fi












