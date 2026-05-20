#!/bin/bash
# Health check script for the deployed service

set -e

# Configuration
APP_NAME="manuales-hogar-ai"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Get load balancer DNS
LB_DNS=$(aws cloudformation describe-stacks \
    --stack-name ${APP_NAME} \
    --region ${AWS_REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
    --output text)

if [ -z "$LB_DNS" ]; then
    echo "Error: Could not retrieve load balancer DNS"
    exit 1
fi

echo "Checking health of service at http://${LB_DNS}..."

# Check health endpoint
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://${LB_DNS}/api/v1/health)

if [ "$HEALTH_RESPONSE" == "200" ]; then
    echo "✅ Service is healthy (HTTP $HEALTH_RESPONSE)"
    
    # Get detailed health info
    echo ""
    echo "Health check details:"
    curl -s http://${LB_DNS}/api/v1/health | jq .
    
    exit 0
else
    echo "❌ Service is unhealthy (HTTP $HEALTH_RESPONSE)"
    exit 1
fi




