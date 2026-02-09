#!/bin/bash

# Health Check Script
# ===================

ENDPOINT="${1:-http://localhost:8000/health}"
MAX_RETRIES=5
RETRY_DELAY=2

echo "🔍 Checking health endpoint: $ENDPOINT"

for i in $(seq 1 $MAX_RETRIES); do
    response=$(curl -s -o /dev/null -w "%{http_code}" "$ENDPOINT" || echo "000")
    
    if [ "$response" = "200" ]; then
        echo "✅ Service is healthy (HTTP $response)"
        curl -s "$ENDPOINT" | python -m json.tool
        exit 0
    else
        echo "⚠️  Attempt $i/$MAX_RETRIES failed (HTTP $response)"
        if [ $i -lt $MAX_RETRIES ]; then
            sleep $RETRY_DELAY
        fi
    fi
done

echo "❌ Health check failed after $MAX_RETRIES attempts"
exit 1
















