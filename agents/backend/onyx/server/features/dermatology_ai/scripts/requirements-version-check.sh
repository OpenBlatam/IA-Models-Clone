#!/bin/bash
# ============================================================================
# Version Check
# Checks if requirements versions are compatible with each other
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Version Compatibility Check"
echo "=========================================="
echo ""

# Check for conflicting version requirements
echo -e "${BLUE}Checking for version conflicts...${NC}"
echo ""

conflicts=0

# Compare requirements.txt with other files
if [ -f "requirements.txt" ]; then
    for file in requirements-optimized.txt requirements-dev.txt; do
        if [ -f "$file" ]; then
            echo -e "${BLUE}Comparing requirements.txt with $file...${NC}"
            
            # Extract packages and versions
            while IFS= read -r line; do
                if [[ $line =~ ^[a-zA-Z] ]] && [[ ! $line =~ ^# ]]; then
                    pkg1=$(echo "$line" | sed 's/[>=<].*//' | sed 's/\[.*\]//')
                    ver1=$(echo "$line" | sed 's/.*[>=<]//' | awk '{print $1}')
                    
                    # Check in other file
                    match=$(grep "^$pkg1" "$file" 2>/dev/null || true)
                    if [ -n "$match" ]; then
                        ver2=$(echo "$match" | sed 's/.*[>=<]//' | awk '{print $1}')
                        if [ "$ver1" != "$ver2" ] && [ -n "$ver1" ] && [ -n "$ver2" ]; then
                            echo -e "  ${YELLOW}⚠ Version conflict: $pkg1${NC}"
                            echo "    requirements.txt: $ver1"
                            echo "    $file: $ver2"
                            ((conflicts++))
                        fi
                    fi
                fi
            done < requirements.txt
            echo ""
        fi
    done
fi

if [ $conflicts -eq 0 ]; then
    echo -e "${GREEN}✓ No version conflicts found${NC}"
else
    echo -e "${YELLOW}⚠ Found $conflicts potential conflicts${NC}"
    echo "💡 Recommendation: Review and align versions"
fi
echo ""

# Check Python version requirements
echo -e "${BLUE}Checking Python version compatibility...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Current Python: $python_version"

# Check if packages require specific Python versions
python_required="3.10"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${GREEN}✓ Python version is compatible${NC}"
else
    echo -e "${RED}✗ Python 3.10+ required${NC}"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}Version check completed${NC}"
echo "=========================================="



