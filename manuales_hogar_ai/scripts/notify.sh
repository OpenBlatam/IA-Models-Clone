#!/bin/bash
# Notification script - Send notifications about service status
# Usage: ./scripts/notify.sh [--webhook URL] [--email EMAIL]

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

WEBHOOK_URL=""
EMAIL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --webhook)
            WEBHOOK_URL="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

API_URL="${API_URL:-http://localhost:8000}"

# Check service status
if curl -f "$API_URL/api/v1/health" > /dev/null 2>&1; then
    STATUS="healthy"
    MESSAGE="✅ Manuales Hogar AI is running and healthy"
    COLOR="good"
else
    STATUS="unhealthy"
    MESSAGE="❌ Manuales Hogar AI is not responding"
    COLOR="danger"
fi

# Get additional info
HEALTH_DATA=$(curl -s "$API_URL/api/v1/health" 2>/dev/null || echo "{}")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Send webhook notification
if [ -n "$WEBHOOK_URL" ]; then
    echo -e "${BLUE}Sending webhook notification...${NC}"
    
    PAYLOAD=$(cat <<EOF
{
    "text": "Manuales Hogar AI Status",
    "attachments": [{
        "color": "$COLOR",
        "fields": [{
            "title": "Status",
            "value": "$STATUS",
            "short": true
        }, {
            "title": "Time",
            "value": "$TIMESTAMP",
            "short": true
        }, {
            "title": "Message",
            "value": "$MESSAGE",
            "short": false
        }]
    }]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
        --data "$PAYLOAD" \
        "$WEBHOOK_URL" > /dev/null 2>&1
    
    echo -e "${GREEN}✅ Webhook notification sent${NC}"
fi

# Send email notification
if [ -n "$EMAIL" ] && command -v mail &> /dev/null; then
    echo -e "${BLUE}Sending email notification...${NC}"
    
    SUBJECT="Manuales Hogar AI Status - $STATUS"
    BODY="Status: $STATUS\nTime: $TIMESTAMP\nMessage: $MESSAGE"
    
    echo -e "$BODY" | mail -s "$SUBJECT" "$EMAIL"
    
    echo -e "${GREEN}✅ Email notification sent${NC}"
fi

# Print status
echo ""
echo "$MESSAGE"
echo "Time: $TIMESTAMP"




