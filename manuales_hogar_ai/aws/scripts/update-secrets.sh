#!/bin/bash
# Script to update secrets in AWS Secrets Manager

set -e

# Configuration
APP_NAME="manuales-hogar-ai"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Update OpenRouter API key
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "Updating OpenRouter API key..."
    aws secretsmanager update-secret \
        --secret-id ${APP_NAME}/openrouter-api-key \
        --secret-string "{\"api_key\": \"${OPENROUTER_API_KEY}\"}" \
        --region ${AWS_REGION}
    echo "OpenRouter API key updated successfully!"
else
    echo "OPENROUTER_API_KEY not set, skipping update"
fi

# Update database password (optional)
if [ -n "$DB_PASSWORD" ]; then
    echo "Updating database password..."
    DB_SECRET=$(aws secretsmanager get-secret-value \
        --secret-id ${APP_NAME}/db-credentials \
        --region ${AWS_REGION} \
        --query 'SecretString' \
        --output text)
    
    DB_USER=$(echo $DB_SECRET | jq -r '.username')
    
    aws secretsmanager update-secret \
        --secret-id ${APP_NAME}/db-credentials \
        --secret-string "{\"username\": \"${DB_USER}\", \"password\": \"${DB_PASSWORD}\"}" \
        --region ${AWS_REGION}
    echo "Database password updated successfully!"
    
    # Restart ECS service to pick up new credentials
    echo "Restarting ECS service to apply new credentials..."
    aws ecs update-service \
        --cluster ${APP_NAME}-cluster \
        --service ${APP_NAME}-service \
        --force-new-deployment \
        --region ${AWS_REGION} > /dev/null
    echo "ECS service restart initiated"
else
    echo "DB_PASSWORD not set, skipping update"
fi

echo "Secrets update completed!"




