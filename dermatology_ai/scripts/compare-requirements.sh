#!/bin/bash
# ============================================================================
# Compare Requirements Files
# Shows differences between requirements files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Requirements File Comparison"
echo "=========================================="
echo ""

# Function to extract package names
extract_packages() {
    grep -E "^[a-zA-Z]" "$1" | grep -v "^#" | sed 's/[>=<].*//' | sed 's/\[.*\]//' | sort
}

# Compare files
compare_files() {
    file1="$1"
    file2="$2"
    
    if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
        return
    fi
    
    echo -e "${BLUE}Comparing: $file1 vs $file2${NC}"
    
    # Packages only in file1
    only_in_1=$(comm -23 <(extract_packages "$file1") <(extract_packages "$file2"))
    # Packages only in file2
    only_in_2=$(comm -13 <(extract_packages "$file1") <(extract_packages "$file2"))
    # Common packages
    common=$(comm -12 <(extract_packages "$file1") <(extract_packages "$file2"))
    
    echo "  Common packages: $(echo "$common" | wc -l)"
    echo "  Only in $file1: $(echo "$only_in_1" | wc -l)"
    echo "  Only in $file2: $(echo "$only_in_2" | wc -l)"
    
    if [ -n "$only_in_1" ]; then
        echo -e "  ${YELLOW}Only in $file1:${NC}"
        echo "$only_in_1" | head -10 | sed 's/^/    - /'
        if [ $(echo "$only_in_1" | wc -l) -gt 10 ]; then
            echo "    ... and $(( $(echo "$only_in_1" | wc -l) - 10 )) more"
        fi
    fi
    
    if [ -n "$only_in_2" ]; then
        echo -e "  ${YELLOW}Only in $file2:${NC}"
        echo "$only_in_2" | head -10 | sed 's/^/    - /'
        if [ $(echo "$only_in_2" | wc -l) -gt 10 ]; then
            echo "    ... and $(( $(echo "$only_in_2" | wc -l) - 10 )) more"
        fi
    fi
    
    echo ""
}

# Main comparisons
compare_files "requirements.txt" "requirements-optimized.txt"
compare_files "requirements.txt" "requirements-dev.txt"
compare_files "requirements-optimized.txt" "requirements-minimal.txt"
compare_files "requirements.txt" "requirements-gpu.txt"
compare_files "requirements.txt" "requirements-docker.txt"

echo "=========================================="
echo -e "${GREEN}Comparison complete${NC}"



