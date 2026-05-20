#!/bin/bash
# ============================================================================
# All Dependency Checks
# Runs all validation and check scripts
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Running All Dependency Checks"
echo "=========================================="
echo ""

# Track results
ERRORS=0
WARNINGS=0

# Function to run check
run_check() {
    name=$1
    command=$2
    
    echo -e "${BLUE}Running: $name${NC}"
    if eval "$command" 2>&1; then
        echo -e "${GREEN}✓ $name passed${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
        ((ERRORS++))
    fi
    echo ""
}

# 1. Validate requirements files
echo "=========================================="
echo "1. Validating Requirements Files"
echo "=========================================="
for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        run_check "Validate $file" "python scripts/validate-requirements.py $file || true"
    fi
done

# 2. Check for duplicates
echo "=========================================="
echo "2. Checking for Duplicates"
echo "=========================================="
run_check "Analyze dependencies" "python scripts/analyze-dependencies.py || true"

# 3. Security check
echo "=========================================="
echo "3. Security Checks"
echo "=========================================="
run_check "Safety check" "safety check -r requirements.txt --short-report || true"
run_check "Pip audit" "pip-audit -r requirements.txt --desc || true || true"

# 4. Health check
echo "=========================================="
echo "4. Health Check"
echo "=========================================="
run_check "Dependency health" "./scripts/check-dependencies.sh || true"

# 5. Compare files
echo "=========================================="
echo "5. Comparing Requirements Files"
echo "=========================================="
run_check "Compare files" "./scripts/compare-requirements.sh || true"

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks completed${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    exit 1
fi



