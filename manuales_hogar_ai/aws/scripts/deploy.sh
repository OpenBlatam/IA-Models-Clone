#!/bin/bash
# Deployment script for Manuales Hogar AI on AWS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="manuales-hogar-ai"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
ECR_REPOSITORY="${ECR_REPOSITORY:-${APP_NAME}}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-dev}"

# Check required environment variables
check_env() {
    echo -e "${YELLOW}Checking environment variables...${NC}"
    
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo -e "${RED}Error: AWS_ACCOUNT_ID is not set${NC}"
        exit 1
    fi
    
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo -e "${YELLOW}Warning: OPENROUTER_API_KEY is not set. It will need to be set in AWS Secrets Manager.${NC}"
    fi
    
    echo -e "${GREEN}Environment check passed${NC}"
}

# Build Docker image
build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    cd "$(dirname "$0")/../.."
    docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
    echo -e "${GREEN}Docker image built successfully${NC}"
}

# Push to ECR
push_to_ecr() {
    echo -e "${YELLOW}Pushing image to ECR...${NC}"
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} || \
        aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION}
    
    # Tag and push
    docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
    
    echo -e "${GREEN}Image pushed to ECR successfully${NC}"
}

# Deploy CDK stack
deploy_cdk() {
    echo -e "${YELLOW}Deploying CDK stack...${NC}"
    cd "$(dirname "$0")/../infrastructure"
    
    # Install CDK dependencies
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    
    # Bootstrap CDK (if needed)
    cdk bootstrap aws://${AWS_ACCOUNT_ID}/${AWS_REGION} || true
    
    # Deploy
    export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
    export DB_USERNAME="${DB_USERNAME:-admin}"
    export DB_PASSWORD="${DB_PASSWORD:-}"
    export ENVIRONMENT="${ENVIRONMENT}"
    
    cdk deploy --require-approval never \
        --context region=${AWS_REGION} \
        --context account=${AWS_ACCOUNT_ID}
    
    echo -e "${GREEN}CDK stack deployed successfully${NC}"
}

# Update ECS service with new image
update_ecs_service() {
    echo -e "${YELLOW}Updating ECS service...${NC}"
    
    CLUSTER_NAME="${APP_NAME}-cluster"
    SERVICE_NAME="${APP_NAME}-service"
    IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
    
    # Get task definition
    TASK_DEF=$(aws ecs describe-task-definition \
        --task-definition ${APP_NAME}-task-def \
        --region ${AWS_REGION} \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    # Update task definition with new image
    aws ecs update-service \
        --cluster ${CLUSTER_NAME} \
        --service ${SERVICE_NAME} \
        --task-definition ${TASK_DEF} \
        --force-new-deployment \
        --region ${AWS_REGION} > /dev/null
    
    echo -e "${GREEN}ECS service update initiated${NC}"
    echo -e "${YELLOW}Waiting for service to stabilize...${NC}"
    
    aws ecs wait services-stable \
        --cluster ${CLUSTER_NAME} \
        --services ${SERVICE_NAME} \
        --region ${AWS_REGION}
    
    echo -e "${GREEN}ECS service updated successfully${NC}"
}

# Main deployment flow
main() {
    echo -e "${GREEN}Starting deployment of ${APP_NAME}...${NC}"
    
    check_env
    build_image
    push_to_ecr
    deploy_cdk
    update_ecs_service
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${YELLOW}Get the load balancer DNS from CDK outputs${NC}"
}

# Run main function
main "$@"




