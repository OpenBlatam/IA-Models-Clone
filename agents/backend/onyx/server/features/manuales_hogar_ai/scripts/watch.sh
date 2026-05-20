#!/bin/bash
# Watch script - Watch for changes and auto-restart
# Usage: ./scripts/watch.sh

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}👀 Watching for changes...${NC}"
echo "Press Ctrl+C to stop"
echo ""

LAST_RESTART=$(date +%s)
RESTART_COOLDOWN=10  # Minimum seconds between restarts

# Function to restart services
restart_services() {
    local current_time=$(date +%s)
    local time_since_restart=$((current_time - LAST_RESTART))
    
    if [ $time_since_restart -lt $RESTART_COOLDOWN ]; then
        echo -e "${YELLOW}⏳ Cooldown active, skipping restart...${NC}"
        return
    fi
    
    echo -e "${BLUE}🔄 Restarting services...${NC}"
    docker-compose restart app
    LAST_RESTART=$(date +%s)
    echo -e "${GREEN}✅ Services restarted${NC}"
    echo ""
}

# Watch for file changes
if command -v inotifywait &> /dev/null; then
    # Linux with inotify
    echo "Using inotifywait..."
    inotifywait -m -r -e modify,create,delete \
        --exclude '\.(git|pyc|pyo|log)' \
        . | while read path action file; do
        echo -e "${YELLOW}📝 Change detected: $file${NC}"
        restart_services
    done
elif command -v fswatch &> /dev/null; then
    # Mac with fswatch
    echo "Using fswatch..."
    fswatch -o . | while read f; do
        echo -e "${YELLOW}📝 Change detected${NC}"
        restart_services
    done
else
    echo -e "${YELLOW}⚠️  No file watcher available (inotifywait or fswatch)${NC}"
    echo "Falling back to polling mode..."
    
    # Polling mode
    LAST_CHECK=$(find . -type f -name "*.py" -newermt "$(date -d '1 minute ago' 2>/dev/null || date -v-1M 2>/dev/null)" 2>/dev/null | head -1)
    
    while true; do
        sleep 5
        CURRENT_CHECK=$(find . -type f -name "*.py" -newermt "$(date -d '1 minute ago' 2>/dev/null || date -v-1M 2>/dev/null)" 2>/dev/null | head -1)
        
        if [ "$CURRENT_CHECK" != "$LAST_CHECK" ] && [ -n "$CURRENT_CHECK" ]; then
            echo -e "${YELLOW}📝 Change detected: $CURRENT_CHECK${NC}"
            restart_services
            LAST_CHECK="$CURRENT_CHECK"
        fi
    done
fi




