#!/bin/bash
# Health check script with detailed information

API_URL="${API_URL:-http://localhost:8000}"

echo "🏥 Health Check - Manuales Hogar AI"
echo "===================================="
echo ""

# Check if service is running
if ! curl -f "${API_URL}/api/v1/health" > /dev/null 2>&1; then
    echo "❌ Service is not responding"
    echo ""
    echo "Check if services are running:"
    echo "  docker-compose ps"
    echo ""
    echo "View logs:"
    echo "  docker-compose logs app"
    exit 1
fi

# Get health status
HEALTH=$(curl -s "${API_URL}/api/v1/health")

echo "✅ Service is healthy"
echo ""
echo "Health Status:"
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
echo ""

# Check other endpoints
echo "Testing endpoints:"
echo ""

# Root endpoint
if curl -f "${API_URL}/" > /dev/null 2>&1; then
    echo "✅ Root endpoint: OK"
else
    echo "❌ Root endpoint: FAILED"
fi

# Docs endpoint
if curl -f "${API_URL}/docs" > /dev/null 2>&1; then
    echo "✅ API Docs: OK"
else
    echo "❌ API Docs: FAILED"
fi

# Models endpoint
if curl -f "${API_URL}/api/v1/models" > /dev/null 2>&1; then
    echo "✅ Models endpoint: OK"
else
    echo "❌ Models endpoint: FAILED"
fi

echo ""




