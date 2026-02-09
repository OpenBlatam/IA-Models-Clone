#!/bin/bash
# Deployment script for Robot Movement AI on AWS
# This script handles building, pushing, and deploying the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="robot-movement-ai"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"
ECS_CLUSTER="${PROJECT_NAME}-cluster"
ECS_SERVICE="${PROJECT_NAME}-service"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$(dirname "$0")/.."
    docker build -f aws/Dockerfile -t ${PROJECT_NAME}:latest .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Login to ECR
ecr_login() {
    log_info "Logging in to ECR..."
    
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin ${ECR_REPO}
    
    if [ $? -eq 0 ]; then
        log_info "ECR login successful"
    else
        log_error "Failed to login to ECR"
        exit 1
    fi
}

# Create ECR repository if it doesn't exist
create_ecr_repo() {
    log_info "Checking ECR repository..."
    
    if ! aws ecr describe-repositories --repository-names ${PROJECT_NAME} --region ${AWS_REGION} &> /dev/null; then
        log_info "Creating ECR repository..."
        aws ecr create-repository --repository-name ${PROJECT_NAME} --region ${AWS_REGION}
        log_info "ECR repository created"
    else
        log_info "ECR repository already exists"
    fi
}

# Tag and push image
push_image() {
    log_info "Tagging and pushing image to ECR..."
    
    docker tag ${PROJECT_NAME}:latest ${ECR_REPO}:latest
    docker tag ${PROJECT_NAME}:latest ${ECR_REPO}:$(date +%Y%m%d-%H%M%S)
    
    docker push ${ECR_REPO}:latest
    docker push ${ECR_REPO}:$(date +%Y%m%d-%H%M%S)
    
    if [ $? -eq 0 ]; then
        log_info "Image pushed successfully"
    else
        log_error "Failed to push image"
        exit 1
    fi
}

# Update ECS service
update_ecs_service() {
    log_info "Updating ECS service..."
    
    aws ecs update-service \
        --cluster ${ECS_CLUSTER} \
        --service ${ECS_SERVICE} \
        --force-new-deployment \
        --region ${AWS_REGION} > /dev/null
    
    if [ $? -eq 0 ]; then
        log_info "ECS service update initiated"
        log_info "Waiting for service to stabilize..."
        aws ecs wait services-stable \
            --cluster ${ECS_CLUSTER} \
            --services ${ECS_SERVICE} \
            --region ${AWS_REGION}
        log_info "Service is stable"
    else
        log_error "Failed to update ECS service"
        exit 1
    fi
}

# Deploy using SAM (for Lambda)
deploy_sam() {
    log_info "Deploying with AWS SAM..."
    
    cd "$(dirname "$0")"
    
    sam build
    sam deploy --guided
    
    if [ $? -eq 0 ]; then
        log_info "SAM deployment successful"
    else
        log_error "SAM deployment failed"
        exit 1
    fi
}

# Main deployment function
main() {
    local deployment_type="${1:-ecs}"
    
    log_info "Starting deployment for ${PROJECT_NAME}"
    log_info "Deployment type: ${deployment_type}"
    log_info "AWS Region: ${AWS_REGION}"
    log_info "AWS Account ID: ${AWS_ACCOUNT_ID}"
    
    check_prerequisites
    
    case ${deployment_type} in
        ecs)
            build_image
            create_ecr_repo
            ecr_login
            push_image
            update_ecs_service
            log_info "ECS deployment completed successfully!"
            ;;
        lambda|sam)
            deploy_sam
            log_info "Lambda deployment completed successfully!"
            ;;
        *)
            log_error "Unknown deployment type: ${deployment_type}"
            log_info "Usage: $0 [ecs|lambda|sam]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"















