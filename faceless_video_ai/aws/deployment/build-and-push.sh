#!/bin/bash
# Build and push Docker image to ECR

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-}
ECR_REPOSITORY="faceless-video-ai"
IMAGE_TAG=${IMAGE_TAG:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Get AWS Account ID if not set
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${YELLOW}Getting AWS Account ID...${NC}"
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo -e "${RED}Error: Could not get AWS Account ID. Make sure AWS credentials are configured.${NC}"
        exit 1
    fi
    echo -e "${GREEN}AWS Account ID: $AWS_ACCOUNT_ID${NC}"
fi

# ECR Repository URI
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building and Pushing Docker Image${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Repository: ${ECR_URI}"
echo -e "Tag: ${IMAGE_TAG}"
echo -e "Region: ${AWS_REGION}"
echo ""

# Login to ECR
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# Check if repository exists, create if not
echo -e "${YELLOW}Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} &> /dev/null; then
    echo -e "${YELLOW}Repository does not exist. Creating...${NC}"
    aws ecr create-repository \
        --repository-name ${ECR_REPOSITORY} \
        --region ${AWS_REGION} \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo -e "${GREEN}Repository created successfully${NC}"
else
    echo -e "${GREEN}Repository exists${NC}"
fi

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
cd "$(dirname "$0")/../.."
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} -f Dockerfile .

# Tag image for ECR
echo -e "${YELLOW}Tagging image...${NC}"
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}:latest

# Push image to ECR
echo -e "${YELLOW}Pushing image to ECR...${NC}"
docker push ${ECR_URI}:${IMAGE_TAG}
docker push ${ECR_URI}:latest

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Successfully pushed image to ECR${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Image URI: ${ECR_URI}:${IMAGE_TAG}"
echo -e "Latest URI: ${ECR_URI}:latest"
echo ""

# Display image details
echo -e "${YELLOW}Image details:${NC}"
aws ecr describe-images \
    --repository-name ${ECR_REPOSITORY} \
    --region ${AWS_REGION} \
    --image-ids imageTag=${IMAGE_TAG} \
    --query 'imageDetails[0].[imageTags, imagePushedAt, imageSizeInBytes]' \
    --output table




