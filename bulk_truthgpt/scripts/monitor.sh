#!/bin/bash

# Monitoring Script
# =================

ENDPOINT="${1:-http://localhost:8000/health}"
INTERVAL="${2:-30}"
MAX_ITERATIONS="${3:-0}"

echo "📊 Starting monitoring (every ${INTERVAL}s)"
echo "Endpoint: $ENDPOINT"
echo ""

iteration=0

while true; do
    iteration=$((iteration + 1))
    
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    response=$(curl -s "$ENDPOINT" 2>/dev/null)
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "$ENDPOINT" 2>/dev/null)
    
    if [ "$status_code" = "200" ]; then
        echo "[$timestamp] ✅ Health OK"
        
        # Extract key metrics if available
        if command -v jq &> /dev/null; then
            echo "$response" | jq -r '.performance.cache.hit_rate_percent, .performance.memory.rss_mb' 2>/dev/null | while read metric; do
                if [ -n "$metric" ]; then
                    echo "  Metric: $metric"
                fi
            done
        fi
    else
        echo "[$timestamp] ❌ Health FAILED (HTTP $status_code)"
    fi
    
    if [ "$MAX_ITERATIONS" -gt 0 ] && [ "$iteration" -ge "$MAX_ITERATIONS" ]; then
        break
    fi
    
    sleep "$INTERVAL"
done

echo ""
echo "📊 Monitoring stopped"
















