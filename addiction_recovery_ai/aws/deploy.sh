#!/bin/bash
# Deployment script for AWS Serverless Application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
STACK_NAME="${STACK_NAME:-addiction-recovery-ai}"
ENVIRONMENT="${ENVIRONMENT:-production}"
REGION="${AWS_REGION:-us-east-1}"

echo -e "${GREEN}Deploying Addiction Recovery AI to AWS...${NC}"
echo "Stack Name: $STACK_NAME"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo -e "${YELLOW}SAM CLI is not installed. Installing...${NC}"
    # Install SAM CLI (macOS/Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install aws-sam-cli
    else
        pip install aws-sam-cli
    fi
fi

# Build Lambda layer dependencies
echo -e "${GREEN}Building Lambda layer...${NC}"
mkdir -p layer/python
pip install -r requirements.txt -t layer/python/
# Remove unnecessary files to reduce size
find layer/python -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find layer/python -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true

# Build SAM application
echo -e "${GREEN}Building SAM application...${NC}"
sam build \
    --template-file aws/sam_template.yaml \
    --build-dir .sam-build

# Deploy SAM application
echo -e "${GREEN}Deploying SAM application...${NC}"
sam deploy \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides Environment="$ENVIRONMENT" \
    --confirm-changeset \
    --resolve-s3

# Get API endpoint
echo -e "${GREEN}Getting API endpoint...${NC}"
API_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
    --output text)

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${GREEN}API URL: $API_URL${NC}"
echo ""
echo "Test the API:"
echo "curl $API_URL/"















