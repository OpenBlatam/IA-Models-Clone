#!/bin/bash
# ============================================================================
# Size Analysis
# Analyzes the size impact of requirements files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Requirements Size Analysis"
echo "=========================================="
echo ""

# Analyze each file
echo -e "${BLUE}File Size Analysis:${NC}"
echo ""

for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        # File size
        file_size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "N/A")
        
        # Package count
        package_count=$(grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | wc -l)
        
        # Line count
        line_count=$(wc -l < "$file")
        
        # Estimate installation size (rough)
        if [ "$package_count" -gt 0 ]; then
            # Very rough estimate: ~10MB per package average
            estimated_mb=$((package_count * 10))
            if [ "$estimated_mb" -gt 1000 ]; then
                estimated_size="${estimated_mb}MB (~$((estimated_mb / 1000))GB)"
            else
                estimated_size="${estimated_mb}MB"
            fi
        else
            estimated_size="N/A"
        fi
        
        echo "  $file:"
        echo "    File size: $file_size"
        echo "    Packages: $package_count"
        echo "    Lines: $line_count"
        echo "    Estimated install: $estimated_size"
        echo ""
    fi
done

# Compare sizes
echo -e "${BLUE}Size Comparison:${NC}"
echo ""

if [ -f "requirements.txt" ] && [ -f "requirements-optimized.txt" ]; then
    main_count=$(grep -E "^[a-zA-Z]" requirements.txt | grep -v "^#" | wc -l)
    opt_count=$(grep -E "^[a-zA-Z]" requirements-optimized.txt | grep -v "^#" | wc -l)
    
    reduction=$((main_count - opt_count))
    reduction_pct=$((reduction * 100 / main_count))
    
    echo "  requirements.txt: $main_count packages"
    echo "  requirements-optimized.txt: $opt_count packages"
    echo "  Reduction: $reduction packages ($reduction_pct%)"
    echo ""
    
    if [ "$reduction_pct" -gt 50 ]; then
        echo -e "  ${GREEN}✓ Significant size reduction with optimized version${NC}"
    fi
fi

echo "=========================================="
echo -e "${GREEN}Analysis completed${NC}"
echo "=========================================="



