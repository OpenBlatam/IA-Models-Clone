#!/bin/bash
# Script to manually scale ECS service

set -e

# Configuration
APP_NAME="manuales-hogar-ai"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${APP_NAME}-cluster"
SERVICE_NAME="${APP_NAME}-service"

# Get desired count from argument or use default
DESIRED_COUNT="${1:-2}"

if ! [[ "$DESIRED_COUNT" =~ ^[0-9]+$ ]]; then
    echo "Error: Desired count must be a number"
    exit 1
fi

echo "Scaling ${SERVICE_NAME} to ${DESIRED_COUNT} tasks..."

aws ecs update-service \
    --cluster ${CLUSTER_NAME} \
    --service ${SERVICE_NAME} \
    --desired-count ${DESIRED_COUNT} \
    --region ${AWS_REGION}

echo "Service scaling initiated. Current status:"
aws ecs describe-services \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'services[0].{DesiredCount:desiredCount,RunningCount:runningCount,PendingCount:pendingCount}' \
    --output table




