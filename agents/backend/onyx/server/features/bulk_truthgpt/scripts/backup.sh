#!/bin/bash

# Backup Script for Bulk TruthGPT
# ================================

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_${TIMESTAMP}"

echo "🔄 Starting backup..."

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup storage
if [ -d "storage" ]; then
    echo "📦 Backing up storage..."
    cp -r storage "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || echo "⚠️  Storage backup failed"
fi

# Backup configuration
if [ -f ".env" ]; then
    echo "⚙️  Backing up configuration..."
    cp .env "$BACKUP_DIR/$BACKUP_NAME/.env.backup" 2>/dev/null || echo "⚠️  Config backup failed"
fi

# Backup logs
if [ -d "logs" ]; then
    echo "📝 Backing up logs..."
    cp -r logs "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || echo "⚠️  Logs backup failed"
fi

# Create backup manifest
cat > "$BACKUP_DIR/$BACKUP_NAME/manifest.txt" << EOF
Backup created: $(date)
Backup name: $BACKUP_NAME
System: Bulk TruthGPT
EOF

# Compress backup
echo "🗜️  Compressing backup..."
cd "$BACKUP_DIR"
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME" 2>/dev/null
rm -rf "$BACKUP_NAME"
cd - > /dev/null

echo "✅ Backup completed: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"

# Cleanup old backups (keep last 7)
echo "🧹 Cleaning up old backups..."
cd "$BACKUP_DIR"
ls -t *.tar.gz 2>/dev/null | tail -n +8 | xargs rm -f 2>/dev/null
cd - > /dev/null

echo "🎉 Backup process completed!"
















