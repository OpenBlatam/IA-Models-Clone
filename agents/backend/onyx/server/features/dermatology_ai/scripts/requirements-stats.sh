#!/bin/bash
# ============================================================================
# Requirements Statistics
# Shows comprehensive statistics about requirements files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Requirements Statistics"
echo "=========================================="
echo ""

# Count packages in each file
echo -e "${BLUE}Package Count by File:${NC}"
echo ""

total_packages=0
for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        count=$(grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | wc -l)
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "N/A")
        lines=$(wc -l < "$file")
        
        echo "  $file:"
        echo "    Packages: $count"
        echo "    Lines: $lines"
        echo "    Size: $size"
        echo ""
        
        total_packages=$((total_packages + count))
    fi
done

echo "Total packages across all files: $total_packages"
echo ""

# Find most common packages
echo -e "${BLUE}Most Common Packages:${NC}"
echo ""
grep -hE "^[a-zA-Z]" requirements*.txt 2>/dev/null | \
    grep -v "^#" | \
    sed 's/[>=<].*//' | \
    sed 's/\[.*\]//' | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{printf "  %-30s %d files\n", $2, $1}'
echo ""

# Check for packages in all files
echo -e "${BLUE}Packages in All Files:${NC}"
echo ""
common_packages=$(comm -12 \
    <(grep -hE "^[a-zA-Z]" requirements.txt requirements-optimized.txt 2>/dev/null | grep -v "^#" | sed 's/[>=<].*//' | sed 's/\[.*\]//' | sort -u) \
    <(grep -hE "^[a-zA-Z]" requirements-dev.txt requirements-minimal.txt 2>/dev/null | grep -v "^#" | sed 's/[>=<].*//' | sed 's/\[.*\]//' | sort -u) 2>/dev/null || echo "")

if [ -n "$common_packages" ]; then
    echo "$common_packages" | head -10 | sed 's/^/  /'
    count=$(echo "$common_packages" | wc -l)
    if [ "$count" -gt 10 ]; then
        echo "  ... and $((count - 10)) more"
    fi
else
    echo "  None found"
fi
echo ""

# File sizes
echo -e "${BLUE}File Sizes:${NC}"
echo ""
for file in requirements*.txt; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "N/A")
        echo "  $file: $size"
    fi
done
echo ""

# Last modified
echo -e "${BLUE}Last Modified:${NC}"
echo ""
for file in requirements*.txt; do
    if [ -f "$file" ]; then
        modified=$(stat -f "%Sm" "$file" 2>/dev/null || stat -c "%y" "$file" 2>/dev/null | cut -d' ' -f1)
        echo "  $file: $modified"
    fi
done
echo ""

echo "=========================================="
echo -e "${GREEN}Statistics complete${NC}"
echo "=========================================="



