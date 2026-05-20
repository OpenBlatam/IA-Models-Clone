#!/bin/bash
# ============================================================================
# Integration Test for Requirements
# Tests that all requirements files work correctly
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

TEMP_VENV=".test-venv"
FAILED=0
PASSED=0

echo "=========================================="
echo "Requirements Integration Test"
echo "=========================================="
echo ""

# Cleanup function
cleanup() {
    if [ -d "$TEMP_VENV" ]; then
        rm -rf "$TEMP_VENV"
    fi
}
trap cleanup EXIT

# Test function
test_requirements_file() {
    file=$1
    if [ ! -f "$file" ]; then
        return
    fi
    
    echo -e "${BLUE}Testing: $file${NC}"
    
    # Create fresh venv
    python3 -m venv "$TEMP_VENV" >/dev/null 2>&1
    source "$TEMP_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip --quiet >/dev/null 2>&1
    
    # Try to install (dry-run)
    if pip install --dry-run -r "$file" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $file - OK"
        ((PASSED++))
    else
        echo -e "  ${RED}✗${NC} $file - FAILED"
        ((FAILED++))
    fi
    
    deactivate
    rm -rf "$TEMP_VENV"
    echo ""
}

# Test all files
for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        test_requirements_file "$file"
    fi
done

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi



