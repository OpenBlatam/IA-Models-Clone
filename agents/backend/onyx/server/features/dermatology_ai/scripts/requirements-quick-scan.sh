#!/bin/bash
# ============================================================================
# Quick Scan
# Fast scan of requirements files for common issues
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "Quick Requirements Scan"
echo "======================="
echo ""

# Quick checks
issues=0

# Check 1: File existence
echo -e "${BLUE}✓ Checking files...${NC}"
for file in requirements.txt requirements-optimized.txt; do
    if [ ! -f "$file" ]; then
        echo -e "  ${RED}✗${NC} $file missing"
        ((issues++))
    else
        echo -e "  ${GREEN}✓${NC} $file exists"
    fi
done
echo ""

# Check 2: Basic format
echo -e "${BLUE}✓ Checking format...${NC}"
for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        # Check for common issues
        if grep -q "  " "$file"; then
            echo -e "  ${YELLOW}⚠${NC} $file has trailing spaces"
        fi
        
        if [ -n "$(tail -c 1 "$file")" ]; then
            echo -e "  ${YELLOW}⚠${NC} $file missing newline at end"
        fi
        
        echo -e "  ${GREEN}✓${NC} $file format OK"
    fi
done
echo ""

# Check 3: Package count
echo -e "${BLUE}✓ Package counts...${NC}"
for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        count=$(grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | wc -l)
        echo -e "  ${GREEN}✓${NC} $file: $count packages"
    fi
done
echo ""

# Summary
echo "======================="
if [ $issues -eq 0 ]; then
    echo -e "${GREEN}✓ Quick scan passed${NC}"
else
    echo -e "${YELLOW}⚠ Found $issues issues${NC}"
fi
echo ""



