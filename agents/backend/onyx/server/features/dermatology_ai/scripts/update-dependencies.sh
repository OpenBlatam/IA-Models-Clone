#!/bin/bash
# ============================================================================
# Dependency Updater Script
# Safely updates dependencies with backup and testing
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REQUIREMENTS_FILE="${1:-requirements.txt}"
BACKUP_DIR=".requirements-backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Dermatology AI - Dependency Updater"
echo "=========================================="
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup current requirements
echo -e "${BLUE}📦 Creating backup...${NC}"
cp "$REQUIREMENTS_FILE" "$BACKUP_DIR/${REQUIREMENTS_FILE}.${TIMESTAMP}"
echo -e "${GREEN}✓ Backup created: $BACKUP_DIR/${REQUIREMENTS_FILE}.${TIMESTAMP}${NC}"
echo ""

# Show outdated packages
echo -e "${BLUE}🔄 Outdated packages:${NC}"
pip list --outdated | head -20
echo ""

# Ask for confirmation
read -p "Update all packages? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Update cancelled"
    exit 0
fi

# Update packages
echo -e "${BLUE}⬆️  Updating packages...${NC}"
pip install --upgrade -r "$REQUIREMENTS_FILE"
echo -e "${GREEN}✓ Packages updated${NC}"
echo ""

# Generate new requirements
echo -e "${BLUE}📝 Generating updated requirements...${NC}"
pip freeze > "$BACKUP_DIR/requirements-frozen.${TIMESTAMP}.txt"
echo -e "${GREEN}✓ Frozen requirements saved${NC}"
echo ""

# Check for security issues
echo -e "${BLUE}🔒 Checking for security issues...${NC}"
if command -v safety &> /dev/null; then
    if safety check -r "$REQUIREMENTS_FILE"; then
        echo -e "${GREEN}✓ No security issues found${NC}"
    else
        echo -e "${RED}✗ Security issues detected${NC}"
        echo "Review the output above"
    fi
else
    echo -e "${YELLOW}⚠ safety not installed, skipping security check${NC}"
fi
echo ""

# Run tests if available
if [ -f "pytest.ini" ] || [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo -e "${BLUE}🧪 Running tests...${NC}"
    if command -v pytest &> /dev/null; then
        if pytest -x --tb=short; then
            echo -e "${GREEN}✓ All tests passed${NC}"
        else
            echo -e "${RED}✗ Tests failed${NC}"
            echo "Consider reverting: cp $BACKUP_DIR/${REQUIREMENTS_FILE}.${TIMESTAMP} $REQUIREMENTS_FILE"
        fi
    else
        echo -e "${YELLOW}⚠ pytest not available${NC}"
    fi
    echo ""
fi

# Summary
echo "=========================================="
echo -e "${GREEN}✓ Update completed${NC}"
echo "=========================================="
echo "Backup location: $BACKUP_DIR/"
echo "Frozen requirements: $BACKUP_DIR/requirements-frozen.${TIMESTAMP}.txt"
echo ""
echo "To revert:"
echo "  cp $BACKUP_DIR/${REQUIREMENTS_FILE}.${TIMESTAMP} $REQUIREMENTS_FILE"
echo "  pip install -r $REQUIREMENTS_FILE"



