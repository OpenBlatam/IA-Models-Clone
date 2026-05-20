#!/bin/bash
# Restore script - Restore database from backup
# Usage: ./scripts/restore.sh <backup_file>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}❌ Usage: ./scripts/restore.sh <backup_file>${NC}"
    echo ""
    echo "Available backups:"
    ls -lh backups/*.sql.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}⚠️  WARNING: This will replace the current database!${NC}"
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}🔄 Restoring database from backup...${NC}"

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}Starting services...${NC}"
    docker-compose up -d postgres
    sleep 5
fi

# Decompress if needed
if [[ "$BACKUP_FILE" == *.gz ]]; then
    echo "Decompressing backup..."
    TEMP_FILE="${BACKUP_FILE%.gz}"
    gunzip -c "$BACKUP_FILE" > "$TEMP_FILE"
    BACKUP_FILE="$TEMP_FILE"
    CLEANUP_TEMP=true
else
    CLEANUP_TEMP=false
fi

# Restore database
echo "Restoring database..."
docker-compose exec -T postgres psql -U postgres -d manuales_hogar < "$BACKUP_FILE" || {
    echo -e "${YELLOW}⚠️  Restore may have failed. Check the output above.${NC}"
}

# Cleanup
if [ "$CLEANUP_TEMP" = true ]; then
    rm -f "$TEMP_FILE"
fi

echo ""
echo -e "${GREEN}✅ Database restored!${NC}"
echo ""
echo "You may need to restart the application:"
echo "  docker-compose restart app"




