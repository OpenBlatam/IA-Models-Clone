#!/bin/bash
# ============================================================================
# Requirements Sync
# Syncs packages between different requirements files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Requirements Sync"
echo "=========================================="
echo ""

# Function to extract packages
extract_packages() {
    grep -E "^[a-zA-Z]" "$1" | grep -v "^#" | sed 's/[>=<].*//' | sed 's/\[.*\]//' | sort
}

# Get packages from main requirements
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

main_packages=$(extract_packages "requirements.txt")
echo -e "${BLUE}Found $(echo "$main_packages" | wc -l) packages in requirements.txt${NC}"
echo ""

# Check other files
for file in requirements-optimized.txt requirements-dev.txt requirements-minimal.txt; do
    if [ -f "$file" ]; then
        file_packages=$(extract_packages "$file")
        missing=$(comm -23 <(echo "$main_packages") <(echo "$file_packages"))
        extra=$(comm -13 <(echo "$main_packages") <(echo "$file_packages"))
        
        echo -e "${BLUE}Analyzing: $file${NC}"
        echo "  Total packages: $(echo "$file_packages" | wc -l)"
        
        if [ -n "$missing" ]; then
            missing_count=$(echo "$missing" | wc -l)
            echo -e "  ${YELLOW}⚠ Missing $missing_count packages from main${NC}"
            echo "$missing" | head -5 | sed 's/^/    - /'
            if [ "$missing_count" -gt 5 ]; then
                echo "    ... and $((missing_count - 5)) more"
            fi
        else
            echo -e "  ${GREEN}✓ All main packages present${NC}"
        fi
        
        if [ -n "$extra" ]; then
            extra_count=$(echo "$extra" | wc -l)
            echo -e "  ${BLUE}ℹ $extra_count extra packages (not in main)${NC}"
        fi
        
        echo ""
    fi
done

echo "=========================================="
echo -e "${GREEN}Sync analysis completed${NC}"
echo "=========================================="



