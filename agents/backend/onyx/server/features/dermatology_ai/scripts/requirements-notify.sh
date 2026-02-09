#!/bin/bash
# ============================================================================
# Requirements Notifications
# Sends notifications about dependency status
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Dependency Status Notifications"
echo "=========================================="
echo ""

# Check outdated
echo -e "${BLUE}Checking outdated packages...${NC}"
outdated=$(pip list --outdated 2>/dev/null | tail -n +3 | wc -l)
if [ "$outdated" -gt 0 ]; then
    echo -e "${YELLOW}⚠ $outdated packages are outdated${NC}"
    echo ""
    echo "Outdated packages:"
    pip list --outdated 2>/dev/null | head -10
    echo ""
    echo "💡 Recommendation: Run 'make update' to update packages"
else
    echo -e "${GREEN}✓ All packages are up to date${NC}"
fi
echo ""

# Check security
echo -e "${BLUE}Checking security...${NC}"
if command -v safety &> /dev/null; then
    if safety check -r requirements.txt --short-report 2>/dev/null; then
        echo -e "${GREEN}✓ No known security vulnerabilities${NC}"
    else
        echo -e "${RED}✗ Security vulnerabilities found!${NC}"
        echo ""
        echo "💡 Recommendation: Run 'make check' for details"
        echo "💡 Recommendation: Update vulnerable packages immediately"
    fi
else
    echo -e "${YELLOW}⚠ safety not installed${NC}"
    echo "💡 Recommendation: Install safety: pip install safety"
fi
echo ""

# Check lock file
echo -e "${BLUE}Checking lock file...${NC}"
if [ -f "requirements-lock.txt" ]; then
    age_days=$(( ($(date +%s) - $(stat -f %m requirements-lock.txt 2>/dev/null || stat -c %Y requirements-lock.txt 2>/dev/null)) / 86400 ))
    if [ "$age_days" -gt 30 ]; then
        echo -e "${YELLOW}⚠ Lock file is $age_days days old${NC}"
        echo "💡 Recommendation: Regenerate with 'make compile'"
    else
        echo -e "${GREEN}✓ Lock file is recent ($age_days days)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No lock file found${NC}"
    echo "💡 Recommendation: Generate with 'make compile'"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary & Recommendations"
echo "=========================================="
echo ""
echo "Quick Actions:"
echo "  make update     - Update packages"
echo "  make check       - Security check"
echo "  make compile     - Generate lock file"
echo "  make audit       - Full audit"
echo ""



