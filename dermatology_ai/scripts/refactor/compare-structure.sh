#!/bin/bash
# ============================================================================
# Compare Structure
# Compares current structure with refactored structure
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=========================================="
echo "Structure Comparison"
echo "=========================================="
echo ""

# Count files in old structure
echo -e "${BLUE}Old Structure (Root):${NC}"
old_services=$(find "$PROJECT_ROOT/services" -maxdepth 1 -name "*.py" 2>/dev/null | wc -l)
old_utils=$(find "$PROJECT_ROOT/utils" -maxdepth 1 -name "*.py" 2>/dev/null | wc -l)
old_docs=$(find "$PROJECT_ROOT" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l)

echo "  Services in root: $old_services"
echo "  Utils in root: $old_utils"
echo "  Docs in root: $old_docs"
echo ""

# Count files in new structure
echo -e "${BLUE}New Structure (Organized):${NC}"
new_services=0
for category in analysis recommendations tracking products ml notifications integrations reporting social shared; do
    count=$(find "$PROJECT_ROOT/services/$category" -name "*.py" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        new_services=$((new_services + count))
        echo "  services/$category/: $count files"
    fi
done

new_utils=0
for category in logging caching validation security performance database async monitoring helpers; do
    count=$(find "$PROJECT_ROOT/utils/$category" -name "*.py" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        new_utils=$((new_utils + count))
        echo "  utils/$category/: $count files"
    fi
done

new_docs=0
for category in architecture dependencies features guides api; do
    count=$(find "$PROJECT_ROOT/docs/$category" -name "*.md" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        new_docs=$((new_docs + count))
        echo "  docs/$category/: $count files"
    fi
done
echo ""

# Comparison
echo "=========================================="
echo "Comparison"
echo "=========================================="
echo ""

if [ "$old_services" -gt 0 ] && [ "$new_services" -gt 0 ]; then
    echo -e "${GREEN}Services:${NC}"
    echo "  Old: $old_services files in root"
    echo "  New: $new_services files organized"
    echo "  Status: $([ "$old_services" -eq 0 ] && echo "✓ Organized" || echo "⚠️  Partially organized")"
    echo ""
fi

if [ "$old_utils" -gt 0 ] && [ "$new_utils" -gt 0 ]; then
    echo -e "${GREEN}Utils:${NC}"
    echo "  Old: $old_utils files in root"
    echo "  New: $new_utils files organized"
    echo "  Status: $([ "$old_utils" -eq 0 ] && echo "✓ Organized" || echo "⚠️  Partially organized")"
    echo ""
fi

if [ "$old_docs" -gt 0 ] && [ "$new_docs" -gt 0 ]; then
    echo -e "${GREEN}Documentation:${NC}"
    echo "  Old: $old_docs files in root"
    echo "  New: $new_docs files organized"
    echo "  Status: $([ "$old_docs" -eq 0 ] && echo "✓ Organized" || echo "⚠️  Partially organized")"
    echo ""
fi

echo "=========================================="



