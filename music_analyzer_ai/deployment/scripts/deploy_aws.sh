#!/bin/bash
# Deployment script for AWS Lambda/ECS

set -e

ENVIRONMENT=${1:-production}
REGION=${2:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Deploying Music Analyzer AI to AWS (Environment: $ENVIRONMENT, Region: $REGION)"

# Check if using Lambda or ECS
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-lambda}

if [ "$DEPLOYMENT_TYPE" == "lambda" ]; then
    echo "Deploying to AWS Lambda..."
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt -t ./package
    
    # Create deployment package
    echo "Creating deployment package..."
    cd package
    zip -r ../lambda-deployment.zip .
    cd ..
    
    # Add application code
    zip -r lambda-deployment.zip . -x "*.git*" -x "*__pycache__*" -x "*.pyc" -x "*tests*" -x "*.md" -x "*deployment/azure*"
    
    # Deploy using AWS CLI or Terraform
    if command -v terraform &> /dev/null; then
        echo "Deploying with Terraform..."
        cd deployment/aws/terraform
        terraform init
        terraform plan -var="environment=$ENVIRONMENT" -var="aws_region=$REGION"
        terraform apply -auto-approve -var="environment=$ENVIRONMENT" -var="aws_region=$REGION"
    else
        echo "Deploying with AWS CLI..."
        aws lambda update-function-code \
            --function-name "music-analyzer-ai-$ENVIRONMENT" \
            --zip-file fileb://lambda-deployment.zip \
            --region $REGION
    fi
    
elif [ "$DEPLOYMENT_TYPE" == "ecs" ]; then
    echo "Deploying to AWS ECS/Fargate..."
    
    # Build Docker image
    echo "Building Docker image..."
    docker build -f deployment/Dockerfile -t music-analyzer-ai:latest .
    
    # Tag for ECR
    ECR_REPO="music-analyzer-ai"
    docker tag music-analyzer-ai:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest
    docker tag music-analyzer-ai:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$ENVIRONMENT
    
    # Login to ECR
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    
    # Push to ECR
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$ENVIRONMENT
    
    # Update ECS service
    aws ecs update-service \
        --cluster music-analyzer-cluster \
        --service music-analyzer-service \
        --force-new-deployment \
        --region $REGION
fi

echo "Deployment complete!"




