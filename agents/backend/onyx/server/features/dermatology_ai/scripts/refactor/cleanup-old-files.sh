#!/bin/bash
# ============================================================================
# Cleanup Old Files
# Removes old duplicate files after refactoring (with confirmation)
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/.refactoring-backup"

echo "=========================================="
echo "Cleanup Old Files After Refactoring"
echo "=========================================="
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will remove old files after refactoring${NC}"
echo "A backup will be created in: $BACKUP_DIR"
echo ""
read -p "Continue? (y/n): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled"
    exit 0
fi

# Create backup
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo -e "${BLUE}Creating backup...${NC}"

# Backup old files before removal
files_to_remove=()

# Check for old service files in root
if [ -d "$PROJECT_ROOT/services" ]; then
    for file in "$PROJECT_ROOT/services"/*.py; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            # Check if file exists in organized structure
            found=false
            for category in analysis recommendations tracking products ml notifications integrations reporting social shared; do
                if [ -f "$PROJECT_ROOT/services/$category/$filename" ]; then
                    found=true
                    break
                fi
            done
            
            if [ "$found" = true ]; then
                files_to_remove+=("$file")
            fi
        fi
    done
fi

# Backup files
if [ ${#files_to_remove[@]} -gt 0 ]; then
    mkdir -p "$BACKUP_DIR/services_$TIMESTAMP"
    for file in "${files_to_remove[@]}"; do
        cp "$file" "$BACKUP_DIR/services_$TIMESTAMP/"
    done
    echo -e "  ${GREEN}✓${NC} Backed up ${#files_to_remove[@]} service files"
fi

# Remove files (with confirmation)
if [ ${#files_to_remove[@]} -gt 0 ]; then
    echo ""
    echo -e "${BLUE}Files to remove:${NC}"
    for file in "${files_to_remove[@]}"; do
        echo "  - $(basename "$file")"
    done
    echo ""
    read -p "Remove these files? (y/n): " confirm_remove
    
    if [[ $confirm_remove =~ ^[Yy]$ ]]; then
        for file in "${files_to_remove[@]}"; do
            rm "$file"
            echo -e "  ${GREEN}✓${NC} Removed $(basename "$file")"
        done
        echo ""
        echo -e "${GREEN}✓ Cleanup completed${NC}"
    else
        echo "Cleanup cancelled"
    fi
else
    echo -e "${GREEN}✓ No duplicate files found${NC}"
fi

echo ""
echo "Backup location: $BACKUP_DIR"



