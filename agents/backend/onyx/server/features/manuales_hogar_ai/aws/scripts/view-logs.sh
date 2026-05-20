#!/bin/bash
# Script to view CloudWatch logs

set -e

# Configuration
APP_NAME="manuales-hogar-ai"
LOG_GROUP="/ecs/${APP_NAME}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Options
FOLLOW="${FOLLOW:-false}"
LINES="${LINES:-100}"

if [ "$FOLLOW" == "true" ]; then
    echo "Following logs from ${LOG_GROUP}..."
    aws logs tail ${LOG_GROUP} --follow --region ${AWS_REGION}
else
    echo "Showing last ${LINES} lines from ${LOG_GROUP}..."
    aws logs tail ${LOG_GROUP} --since 1h --region ${AWS_REGION} | tail -n ${LINES}
fi




