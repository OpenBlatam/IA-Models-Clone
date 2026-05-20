#!/bin/bash
# ============================================================================
# Master Check
# Runs all checks and validations in one command
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Master Requirements Check"
echo "=========================================="
echo ""

TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to run check
run_check() {
    name=$1
    command=$2
    
    ((TOTAL_CHECKS++))
    echo -e "${BLUE}[$TOTAL_CHECKS] $name${NC}"
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓ PASSED${NC}"
        ((PASSED_CHECKS++))
    else
        echo -e "  ${RED}✗ FAILED${NC}"
        ((FAILED_CHECKS++))
    fi
    echo ""
}

# Run all checks
run_check "Quick Scan" "./scripts/requirements-quick-scan.sh"
run_check "Format Validation" "python scripts/validate-requirements.py requirements.txt"
run_check "Health Check" "./scripts/check-dependencies.sh"
run_check "Security Check" "safety check -r requirements.txt --short-report"
run_check "Outdated Check" "pip list --outdated | head -5"
run_check "Size Analysis" "./scripts/requirements-size-analysis.sh"
run_check "Conflict Detection" "python scripts/requirements-dependency-conflicts.py requirements.txt"

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    exit 1
fi



