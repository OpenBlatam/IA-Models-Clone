#!/bin/bash
# ============================================================================
# Development Environment Setup Script
# Sets up complete development environment
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Dermatology AI - Dev Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo -e "${BLUE}🐍 Checking Python version...${NC}"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "Error: Python 3.10+ required"
    exit 1
fi
python3 --version
echo ""

# Create virtual environment
echo -e "${BLUE}📦 Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}🔌 Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install development dependencies
echo -e "${BLUE}📚 Installing development dependencies...${NC}"
pip install -r requirements-dev.txt
echo -e "${GREEN}✓ Development dependencies installed${NC}"
echo ""

# Install pre-commit hooks
echo -e "${BLUE}🪝 Setting up pre-commit hooks...${NC}"
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
else
    echo -e "${YELLOW}⚠ .pre-commit-config.yaml not found${NC}"
fi
echo ""

# Run initial checks
echo -e "${BLUE}🔍 Running initial checks...${NC}"
if command -v ruff &> /dev/null; then
    ruff check . --fix || true
    echo -e "${GREEN}✓ Code formatted${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✓ Development environment ready!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Run tests: make test"
echo "  3. Start server: make serve"
echo "  4. Check code: make lint"
echo ""



