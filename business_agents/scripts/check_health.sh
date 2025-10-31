#!/bin/bash
# Health check script for monitoring

API_URL="${1:-http://localhost:8000}"

echo "🔍 Checking API health at $API_URL..."

# Check liveness
echo -n "Liveness: "
liveness=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/live")
if [ "$liveness" = "200" ]; then
    echo "✅ OK"
else
    echo "❌ FAILED ($liveness)"
    exit 1
fi

# Check readiness
echo -n "Readiness: "
readiness=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/ready")
if [ "$readiness" = "200" ]; then
    echo "✅ OK"
else
    echo "❌ FAILED ($readiness)"
    exit 1
fi

# Check health
echo -n "Health: "
health=$(curl -s "$API_URL/health" | grep -o '"status":"[^"]*"' | head -1)
if echo "$health" | grep -q "healthy"; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    exit 1
fi

echo "✅ All health checks passed!"


