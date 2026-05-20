#!/bin/bash
# ============================================================================
# Check Refactoring Status
# Checks what has been refactored and what needs refactoring
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
echo "Refactoring Status Check"
echo "=========================================="
echo ""

# Check scripts/requirements structure
echo -e "${BLUE}Checking scripts/requirements structure...${NC}"
if [ -d "$PROJECT_ROOT/scripts/requirements" ]; then
    if [ -d "$PROJECT_ROOT/scripts/requirements/analysis" ] && \
       [ -d "$PROJECT_ROOT/scripts/requirements/validation" ] && \
       [ -d "$PROJECT_ROOT/scripts/requirements/management" ] && \
       [ -d "$PROJECT_ROOT/scripts/requirements/utils" ]; then
        echo -e "  ${GREEN}✓${NC} Scripts structure organized"
    else
        echo -e "  ${YELLOW}⚠${NC} Scripts structure incomplete"
    fi
else
    echo -e "  ${RED}✗${NC} Scripts structure not found"
fi
echo ""

# Check docs structure
echo -e "${BLUE}Checking docs structure...${NC}"
if [ -d "$PROJECT_ROOT/docs" ]; then
    if [ -d "$PROJECT_ROOT/docs/architecture" ] && \
       [ -d "$PROJECT_ROOT/docs/dependencies" ] && \
       [ -d "$PROJECT_ROOT/docs/features" ] && \
       [ -d "$PROJECT_ROOT/docs/guides" ] && \
       [ -d "$PROJECT_ROOT/docs/api" ]; then
        echo -e "  ${GREEN}✓${NC} Docs structure organized"
    else
        echo -e "  ${YELLOW}⚠${NC} Docs structure incomplete"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Docs structure not created yet"
fi
echo ""

# Check config structure
echo -e "${BLUE}Checking config structure...${NC}"
if [ -d "$PROJECT_ROOT/config/environments" ]; then
    echo -e "  ${GREEN}✓${NC} Config structure organized"
else
    echo -e "  ${YELLOW}⚠${NC} Config structure not organized yet"
fi
echo ""

# Count files in root
echo -e "${BLUE}Checking root directory...${NC}"
root_files=$(find "$PROJECT_ROOT" -maxdepth 1 -type f -name "*.md" | wc -l)
if [ "$root_files" -gt 10 ]; then
    echo -e "  ${YELLOW}⚠${NC} $root_files documentation files in root (consider moving to docs/)"
else
    echo -e "  ${GREEN}✓${NC} Root directory clean"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "To complete refactoring:"
echo "  1. Run: make refactor-docs"
echo "  2. Run: make refactor-config"
echo "  3. Run: python scripts/refactor/analyze-structure.py"
echo ""



