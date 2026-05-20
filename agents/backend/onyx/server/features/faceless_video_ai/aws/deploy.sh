#!/bin/bash
# AWS Deployment Script for Faceless Video AI

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-}
ECR_REPOSITORY=${ECR_REPOSITORY:-faceless-video-ai}
IMAGE_TAG=${IMAGE_TAG:-latest}
ECS_CLUSTER=${ECS_CLUSTER:-faceless-video-ai-cluster}
ECS_SERVICE=${ECS_SERVICE:-faceless-video-ai-service}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AWS deployment...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Get AWS account ID if not set
if [ -z "$AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo -e "${YELLOW}Using AWS Account ID: $AWS_ACCOUNT_ID${NC}"
fi

# ECR login
echo -e "${GREEN}Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repository if it doesn't exist
echo -e "${GREEN}Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION &> /dev/null; then
    echo -e "${YELLOW}Creating ECR repository...${NC}"
    aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
fi

# Build Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t $ECR_REPOSITORY:$IMAGE_TAG .

# Tag image for ECR
ECR_IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG
docker tag $ECR_REPOSITORY:$IMAGE_TAG $ECR_IMAGE_URI

# Push image to ECR
echo -e "${GREEN}Pushing image to ECR...${NC}"
docker push $ECR_IMAGE_URI

# Update ECS service
if aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE --region $AWS_REGION &> /dev/null; then
    echo -e "${GREEN}Updating ECS service...${NC}"
    aws ecs update-service \
        --cluster $ECS_CLUSTER \
        --service $ECS_SERVICE \
        --force-new-deployment \
        --region $AWS_REGION
    
    echo -e "${GREEN}Waiting for service to stabilize...${NC}"
    aws ecs wait services-stable \
        --cluster $ECS_CLUSTER \
        --services $ECS_SERVICE \
        --region $AWS_REGION
else
    echo -e "${YELLOW}ECS service does not exist. Please create it first using the task definition.${NC}"
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}Image URI: $ECR_IMAGE_URI${NC}"




