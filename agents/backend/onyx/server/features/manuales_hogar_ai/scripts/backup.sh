#!/bin/bash
# Backup script - Backup database and important files
# Usage: ./scripts/backup.sh [--full]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_BACKUP=false

if [ "$1" == "--full" ]; then
    FULL_BACKUP=true
fi

echo -e "${BLUE}💾 Creating backup...${NC}"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Services are not running. Starting services for backup...${NC}"
    docker-compose up -d postgres
    sleep 5
fi

# Backup database
echo -e "${BLUE}Backing up database...${NC}"
DB_BACKUP_FILE="$BACKUP_DIR/db_backup_$TIMESTAMP.sql"
docker-compose exec -T postgres pg_dump -U postgres manuales_hogar > "$DB_BACKUP_FILE"
gzip "$DB_BACKUP_FILE"
echo -e "${GREEN}✅ Database backed up to: ${DB_BACKUP_FILE}.gz${NC}"

# Full backup (includes volumes and config)
if [ "$FULL_BACKUP" = true ]; then
    echo ""
    echo -e "${BLUE}Creating full backup...${NC}"
    
    FULL_BACKUP_FILE="$BACKUP_DIR/full_backup_$TIMESTAMP.tar.gz"
    
    # Backup volumes
    echo "  Backing up volumes..."
    docker run --rm \
        -v manuales_hogar_ai_postgres_data:/data:ro \
        -v "$(pwd)":/backup \
        alpine tar czf "/backup/volumes_backup_$TIMESTAMP.tar.gz" -C /data . 2>/dev/null || true
    
    # Backup configuration
    echo "  Backing up configuration..."
    tar czf "$FULL_BACKUP_FILE" \
        .env \
        docker-compose.yml \
        docker-compose.prod.yml \
        alembic.ini \
        alembic/ \
        config/ \
        2>/dev/null || true
    
    echo -e "${GREEN}✅ Full backup created: $FULL_BACKUP_FILE${NC}"
fi

# Clean old backups (keep last 10)
echo ""
echo -e "${BLUE}Cleaning old backups (keeping last 10)...${NC}"
ls -t "$BACKUP_DIR"/*.gz 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true
echo -e "${GREEN}✅ Old backups cleaned${NC}"

echo ""
echo -e "${GREEN}🎉 Backup complete!${NC}"
echo ""
echo "Backup location: $BACKUP_DIR"
ls -lh "$BACKUP_DIR" | tail -n +2




