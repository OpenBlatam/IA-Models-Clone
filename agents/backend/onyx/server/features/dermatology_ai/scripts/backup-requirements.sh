#!/bin/bash
# ============================================================================
# Backup Requirements Files
# Creates timestamped backups of all requirements files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

BACKUP_DIR=".requirements-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="requirements-backup-${TIMESTAMP}"

echo "=========================================="
echo "Backing up requirements files"
echo "=========================================="
echo ""

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup all requirements files
backed_up=0
for file in requirements*.txt; do
    if [ -f "$file" ]; then
        cp "$file" "${BACKUP_DIR}/${BACKUP_NAME}/"
        echo -e "${BLUE}✓ Backed up: $file${NC}"
        ((backed_up++))
    fi
done

# Backup configuration files
if [ -f "pyproject.toml" ]; then
    cp "pyproject.toml" "${BACKUP_DIR}/${BACKUP_NAME}/"
    echo -e "${BLUE}✓ Backed up: pyproject.toml${NC}"
fi

if [ -f "Makefile" ]; then
    cp "Makefile" "${BACKUP_DIR}/${BACKUP_NAME}/"
    echo -e "${BLUE}✓ Backed up: Makefile${NC}"
fi

# Create manifest
cat > "${BACKUP_DIR}/${BACKUP_NAME}/MANIFEST.txt" << EOF
Requirements Backup
==================
Timestamp: $(date)
Backup Name: ${BACKUP_NAME}
Files Backed Up: ${backed_up}

Files:
$(ls -1 "${BACKUP_DIR}/${BACKUP_NAME}" | grep -v MANIFEST)
EOF

# Compress backup
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"
cd ..

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Backup completed${NC}"
echo "=========================================="
echo "Backup location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "Files backed up: ${backed_up}"
echo ""

# List recent backups
echo "Recent backups:"
ls -lt "${BACKUP_DIR}"/*.tar.gz 2>/dev/null | head -5 | awk '{print "  " $9 " (" $6 " " $7 " " $8 ")"}' || echo "  No previous backups"



