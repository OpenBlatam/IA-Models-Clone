#!/bin/bash
# ============================================================================
# Validate Refactoring
# Validates that refactoring was successful
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=========================================="
echo "Validating Refactoring"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# Check services structure
echo -e "${BLUE}Checking services structure...${NC}"
if [ -d "$PROJECT_ROOT/services" ]; then
    categories=("analysis" "recommendations" "tracking" "products" "ml" "notifications" "integrations" "reporting" "social" "shared")
    for category in "${categories[@]}"; do
        if [ -d "$PROJECT_ROOT/services/$category" ]; then
            count=$(find "$PROJECT_ROOT/services/$category" -name "*.py" | wc -l)
            if [ "$count" -gt 0 ]; then
                echo -e "  ${GREEN}✓${NC} $category/ ($count files)"
            else
                echo -e "  ${YELLOW}⚠${NC} $category/ (empty)"
                ((WARNINGS++))
            fi
        else
            echo -e "  ${YELLOW}⚠${NC} $category/ (missing)"
            ((WARNINGS++))
        fi
    done
else
    echo -e "  ${RED}✗${NC} services/ directory not found"
    ((ERRORS++))
fi
echo ""

# Check utils structure
echo -e "${BLUE}Checking utils structure...${NC}"
if [ -d "$PROJECT_ROOT/utils" ]; then
    categories=("logging" "caching" "validation" "security" "performance" "database" "async" "monitoring" "helpers")
    for category in "${categories[@]}"; do
        if [ -d "$PROJECT_ROOT/utils/$category" ]; then
            count=$(find "$PROJECT_ROOT/utils/$category" -name "*.py" | wc -l)
            if [ "$count" -gt 0 ]; then
                echo -e "  ${GREEN}✓${NC} $category/ ($count files)"
            else
                echo -e "  ${YELLOW}⚠${NC} $category/ (empty)"
                ((WARNINGS++))
            fi
        else
            echo -e "  ${YELLOW}⚠${NC} $category/ (missing)"
            ((WARNINGS++))
        fi
    done
else
    echo -e "  ${RED}✗${NC} utils/ directory not found"
    ((ERRORS++))
fi
echo ""

# Check docs structure
echo -e "${BLUE}Checking docs structure...${NC}"
if [ -d "$PROJECT_ROOT/docs" ]; then
    categories=("architecture" "dependencies" "features" "guides" "api")
    for category in "${categories[@]}"; do
        if [ -d "$PROJECT_ROOT/docs/$category" ]; then
            count=$(find "$PROJECT_ROOT/docs/$category" -name "*.md" 2>/dev/null | wc -l)
            echo -e "  ${GREEN}✓${NC} $category/ ($count files)"
        else
            echo -e "  ${YELLOW}⚠${NC} $category/ (missing)"
            ((WARNINGS++))
        fi
    done
else
    echo -e "  ${YELLOW}⚠${NC} docs/ directory not found"
    ((WARNINGS++))
fi
echo ""

# Check scripts structure
echo -e "${BLUE}Checking scripts structure...${NC}"
if [ -d "$PROJECT_ROOT/scripts/requirements" ]; then
    categories=("analysis" "validation" "management" "utils")
    for category in "${categories[@]}"; do
        if [ -d "$PROJECT_ROOT/scripts/requirements/$category" ]; then
            count=$(find "$PROJECT_ROOT/scripts/requirements/$category" -type f | wc -l)
            echo -e "  ${GREEN}✓${NC} requirements/$category/ ($count files)"
        else
            echo -e "  ${YELLOW}⚠${NC} requirements/$category/ (missing)"
            ((WARNINGS++))
        fi
    done
else
    echo -e "  ${YELLOW}⚠${NC} scripts/requirements/ not found"
    ((WARNINGS++))
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ Refactoring validation passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Refactoring mostly complete (some warnings)${NC}"
    exit 0
else
    echo -e "${RED}✗ Refactoring validation failed${NC}"
    exit 1
fi



