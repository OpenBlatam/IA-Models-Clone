#!/bin/bash
# Export logs script - Export all logs to a file
# Usage: ./scripts/export-logs.sh [--all]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

EXPORT_ALL=false
if [ "$1" == "--all" ]; then
    EXPORT_ALL=true
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs"
EXPORT_FILE="$LOG_DIR/logs_export_$TIMESTAMP.txt"

mkdir -p "$LOG_DIR"

echo -e "${BLUE}📋 Exporting logs...${NC}"
echo ""

# System information
{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "OS: $(uname -s)"
    echo "Kernel: $(uname -r)"
    echo ""
    
    # Docker information
    echo "=== Docker Information ==="
    docker --version 2>/dev/null || echo "Docker not available"
    echo ""
    
    # Service status
    echo "=== Service Status ==="
    docker-compose ps 2>/dev/null || echo "Services not running"
    echo ""
    
    # Application logs
    echo "=== Application Logs ==="
    if [ "$EXPORT_ALL" = true ]; then
        docker-compose logs --no-color 2>/dev/null || echo "No logs available"
    else
        docker-compose logs --tail=100 --no-color 2>/dev/null || echo "No logs available"
    fi
    echo ""
    
    # Database logs
    echo "=== Database Logs ==="
    docker-compose logs --tail=50 postgres --no-color 2>/dev/null || echo "No database logs"
    echo ""
    
    # Redis logs
    echo "=== Redis Logs ==="
    docker-compose logs --tail=50 redis --no-color 2>/dev/null || echo "No Redis logs"
    echo ""
    
    # Health status
    echo "=== Health Status ==="
    curl -s http://localhost:8000/api/v1/health 2>/dev/null || echo "Health check failed"
    echo ""
    
} > "$EXPORT_FILE"

echo -e "${GREEN}✅ Logs exported to: $EXPORT_FILE${NC}"
echo ""
echo "File size: $(du -h "$EXPORT_FILE" | cut -f1)"
echo ""
echo "View logs:"
echo "  cat $EXPORT_FILE"
echo "  less $EXPORT_FILE"




