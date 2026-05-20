#!/bin/bash
# ============================================================================
# Cleanup Requirements
# Removes old backups, cache, and temporary files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Cleaning up requirements files"
echo "=========================================="
echo ""

# Clean pip cache
echo -e "${BLUE}Cleaning pip cache...${NC}"
pip cache purge 2>/dev/null || true
echo -e "${GREEN}✓ Pip cache cleaned${NC}"
echo ""

# Clean old backups (keep last 5)
echo -e "${BLUE}Cleaning old backups...${NC}"
if [ -d ".requirements-backups" ]; then
    backup_count=$(ls -1 .requirements-backups/*.tar.gz 2>/dev/null | wc -l)
    if [ "$backup_count" -gt 5 ]; then
        # Remove oldest backups
        ls -t .requirements-backups/*.tar.gz | tail -n +6 | xargs rm -f
        removed=$((backup_count - 5))
        echo -e "${GREEN}✓ Removed $removed old backups${NC}"
    else
        echo -e "${GREEN}✓ No old backups to remove${NC}"
    fi
else
    echo -e "${GREEN}✓ No backups directory${NC}"
fi
echo ""

# Clean generated files
echo -e "${BLUE}Cleaning generated files...${NC}"
generated_files=(
    "dependency-analysis.json"
    "dependency-visualization.json"
    "dependency-visualization.png"
    "REQUIREMENTS_REPORT.md"
    "bandit-report.json"
    "safety-report.json"
    "*.pyc"
    "__pycache__"
)

cleaned=0
for pattern in "${generated_files[@]}"; do
    if [ -f "$pattern" ] || [ -d "$pattern" ]; then
        rm -rf "$pattern"
        ((cleaned++))
    fi
done

# Clean __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

if [ $cleaned -gt 0 ]; then
    echo -e "${GREEN}✓ Cleaned $cleaned generated files${NC}"
else
    echo -e "${GREEN}✓ No generated files to clean${NC}"
fi
echo ""

# Clean temporary files
echo -e "${BLUE}Cleaning temporary files...${NC}"
temp_files=(
    ".benchmark-venv"
    "requirements-updated.txt"
    "requirements-frozen.*.txt"
)

temp_cleaned=0
for pattern in "${temp_files[@]}"; do
    for file in $pattern; do
        if [ -e "$file" ]; then
            rm -rf "$file"
            ((temp_cleaned++))
        fi
    done
done

if [ $temp_cleaned -gt 0 ]; then
    echo -e "${GREEN}✓ Cleaned $temp_cleaned temporary files${NC}"
else
    echo -e "${GREEN}✓ No temporary files to clean${NC}"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}✓ Cleanup completed${NC}"
echo "=========================================="



