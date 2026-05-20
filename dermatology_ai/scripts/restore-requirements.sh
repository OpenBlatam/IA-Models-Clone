#!/bin/bash
# ============================================================================
# Restore Requirements Files
# Restores requirements files from backup
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

BACKUP_DIR=".requirements-backups"

echo "=========================================="
echo "Restore Requirements Files"
echo "=========================================="
echo ""

# List available backups
if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A ${BACKUP_DIR}/*.tar.gz 2>/dev/null)" ]; then
    echo -e "${RED}No backups found in ${BACKUP_DIR}${NC}"
    exit 1
fi

echo "Available backups:"
echo ""
backups=($(ls -t ${BACKUP_DIR}/*.tar.gz))
for i in "${!backups[@]}"; do
    backup_file="${backups[$i]}"
    backup_name=$(basename "$backup_file" .tar.gz)
    backup_date=$(stat -f "%Sm" "$backup_file" 2>/dev/null || stat -c "%y" "$backup_file" 2>/dev/null | cut -d' ' -f1-2)
    echo "  $((i+1)). $backup_name"
    echo "     Date: $backup_date"
done
echo ""

# Select backup
read -p "Select backup number (1-${#backups[@]}): " selection
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt "${#backups[@]}" ]; then
    echo -e "${RED}Invalid selection${NC}"
    exit 1
fi

selected_backup="${backups[$((selection-1))]}"
backup_name=$(basename "$selected_backup" .tar.gz)

echo ""
echo -e "${YELLOW}Selected: $backup_name${NC}"
read -p "Restore this backup? (y/n): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    exit 0
fi

# Extract backup
echo ""
echo -e "${BLUE}Extracting backup...${NC}"
TEMP_DIR=$(mktemp -d)
tar -xzf "$selected_backup" -C "$TEMP_DIR"

# Restore files
echo -e "${BLUE}Restoring files...${NC}"
restored=0
for file in "$TEMP_DIR/$backup_name"/*.txt; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "./$filename"
        echo -e "${GREEN}✓ Restored: $filename${NC}"
        ((restored++))
    fi
done

# Restore config files
if [ -f "$TEMP_DIR/$backup_name/pyproject.toml" ]; then
    cp "$TEMP_DIR/$backup_name/pyproject.toml" "./pyproject.toml"
    echo -e "${GREEN}✓ Restored: pyproject.toml${NC}"
fi

if [ -f "$TEMP_DIR/$backup_name/Makefile" ]; then
    cp "$TEMP_DIR/$backup_name/Makefile" "./Makefile"
    echo -e "${GREEN}✓ Restored: Makefile${NC}"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Restore completed${NC}"
echo "=========================================="
echo "Files restored: ${restored}"



