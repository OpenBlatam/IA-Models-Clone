#!/bin/bash
# Script to initialize database after deployment

set -e

# Configuration
APP_NAME="manuales-hogar-ai"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Get database endpoint from CloudFormation stack
DB_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name ${APP_NAME} \
    --region ${AWS_REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
    --output text)

# Get database credentials from Secrets Manager
DB_SECRET=$(aws secretsmanager get-secret-value \
    --secret-id ${APP_NAME}/db-credentials \
    --region ${AWS_REGION} \
    --query 'SecretString' \
    --output text)

DB_USER=$(echo $DB_SECRET | jq -r '.username')
DB_PASSWORD=$(echo $DB_SECRET | jq -r '.password')
DB_NAME="manuales_hogar"

# Set DATABASE_URL
export DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_ENDPOINT}:5432/${DB_NAME}"

echo "Running Alembic migrations..."
cd "$(dirname "$0")/../.."
alembic upgrade head

echo "Database initialized successfully!"




