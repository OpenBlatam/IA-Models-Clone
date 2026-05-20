#!/bin/bash
# ============================================================================
# Auto-fix Requirements
# Automatically fixes common issues in requirements files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BACKUP_DIR=".requirements-backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Auto-fix Requirements"
echo "=========================================="
echo ""

# Create backup
mkdir -p "$BACKUP_DIR"
echo -e "${BLUE}Creating backup...${NC}"

for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        cp "$file" "$BACKUP_DIR/${file}.${TIMESTAMP}"
        echo -e "  ${GREEN}✓${NC} Backed up $file"
    fi
done
echo ""

# Fix function
fix_file() {
    file=$1
    if [ ! -f "$file" ]; then
        return
    fi
    
    echo -e "${BLUE}Fixing $file...${NC}"
    
    # Create temp file
    temp_file="${file}.tmp"
    cp "$file" "$temp_file"
    
    # Fix 1: Remove trailing whitespace
    sed -i '' 's/[[:space:]]*$//' "$temp_file" 2>/dev/null || sed -i 's/[[:space:]]*$//' "$temp_file"
    
    # Fix 2: Ensure proper line endings
    dos2unix "$temp_file" 2>/dev/null || true
    
    # Fix 3: Sort packages (optional, commented out to preserve structure)
    # sort -u "$temp_file" > "${temp_file}.sorted"
    # mv "${temp_file}.sorted" "$temp_file"
    
    # Fix 4: Remove empty lines at end
    sed -i '' -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$temp_file" 2>/dev/null || \
    sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$temp_file"
    
    # Replace original if different
    if ! cmp -s "$file" "$temp_file"; then
        mv "$temp_file" "$file"
        echo -e "  ${GREEN}✓${NC} Fixed $file"
    else
        rm "$temp_file"
        echo -e "  ${BLUE}ℹ${NC} No changes needed for $file"
    fi
}

# Fix all files
for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        fix_file "$file"
    fi
done

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Auto-fix completed${NC}"
echo "=========================================="
echo "Backups saved to: $BACKUP_DIR/"
echo ""



