#!/bin/bash
# Health monitor script - Continuous health monitoring with alerts
# Usage: ./scripts/health-monitor.sh [--interval SECONDS] [--webhook URL]

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"
INTERVAL="${INTERVAL:-30}"
WEBHOOK_URL=""
CONSECUTIVE_FAILURES=0
MAX_FAILURES=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --webhook)
            WEBHOOK_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🏥 Health Monitor - Manuales Hogar AI${NC}"
echo "Monitoring: $API_URL"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

send_alert() {
    local message="$1"
    echo -e "${RED}🚨 ALERT: $message${NC}"
    
    if [ -n "$WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Manuales Hogar AI Alert: $message\"}" \
            "$WEBHOOK_URL" > /dev/null 2>&1
    fi
}

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    if curl -f -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
        if [ $CONSECUTIVE_FAILURES -gt 0 ]; then
            echo -e "${GREEN}[$TIMESTAMP] ✅ Service recovered${NC}"
            CONSECUTIVE_FAILURES=0
        else
            echo -e "${GREEN}[$TIMESTAMP] ✅ Healthy${NC}"
        fi
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        echo -e "${RED}[$TIMESTAMP] ❌ Unhealthy (failure $CONSECUTIVE_FAILURES/$MAX_FAILURES)${NC}"
        
        if [ $CONSECUTIVE_FAILURES -ge $MAX_FAILURES ]; then
            send_alert "Service has been unhealthy for $CONSECUTIVE_FAILURES consecutive checks"
        fi
    fi
    
    sleep $INTERVAL
done




