#!/bin/bash
# Script de deployment para AWS
# =============================

set -e  # Salir en error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuración
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
ECR_REPOSITORY=${ECR_REPOSITORY:-cursor-agent-24-7}
IMAGE_TAG=${IMAGE_TAG:-latest}
LAMBDA_FUNCTION_NAME=${LAMBDA_FUNCTION_NAME:-cursor-agent-24-7}
ECS_CLUSTER=${ECS_CLUSTER:-cursor-agent-cluster}
ECS_SERVICE=${ECS_SERVICE:-cursor-agent-service}

echo -e "${GREEN}🚀 Starting AWS deployment...${NC}"

# Verificar AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
    exit 1
fi

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install it first.${NC}"
    exit 1
fi

# Función para deployment en ECS
deploy_ecs() {
    echo -e "${YELLOW}📦 Deploying to ECS...${NC}"
    
    # Login a ECR
    echo "Logging in to ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Crear repositorio ECR si no existe
    if ! aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION &> /dev/null; then
        echo "Creating ECR repository..."
        aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
    fi
    
    # Build y push imagen
    ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG"
    
    echo "Building Docker image..."
    docker build -t $ECR_REPOSITORY:$IMAGE_TAG -f Dockerfile .
    
    echo "Tagging image..."
    docker tag $ECR_REPOSITORY:$IMAGE_TAG $ECR_URI
    
    echo "Pushing image to ECR..."
    docker push $ECR_URI
    
    # Actualizar servicio ECS
    echo "Updating ECS service..."
    aws ecs update-service \
        --cluster $ECS_CLUSTER \
        --service $ECS_SERVICE \
        --force-new-deployment \
        --region $AWS_REGION
    
    echo -e "${GREEN}✅ ECS deployment completed!${NC}"
}

# Función para deployment en Lambda
deploy_lambda() {
    echo -e "${YELLOW}⚡ Deploying to Lambda...${NC}"
    
    # Crear deployment package
    echo "Creating deployment package..."
    mkdir -p .lambda_package
    cp -r . .lambda_package/
    cd .lambda_package
    
    # Instalar dependencias
    echo "Installing dependencies..."
    pip install -r requirements-lambda.txt -t .
    
    # Crear ZIP
    echo "Creating ZIP package..."
    zip -r ../lambda-deployment.zip . -x "*.git*" -x "*.pyc" -x "__pycache__/*"
    cd ..
    
    # Actualizar función Lambda
    echo "Updating Lambda function..."
    aws lambda update-function-code \
        --function-name $LAMBDA_FUNCTION_NAME \
        --zip-file fileb://lambda-deployment.zip \
        --region $AWS_REGION
    
    # Limpiar
    rm -rf .lambda_package lambda-deployment.zip
    
    echo -e "${GREEN}✅ Lambda deployment completed!${NC}"
}

# Función para deployment con Docker en Lambda (container image)
deploy_lambda_container() {
    echo -e "${YELLOW}🐳 Deploying Lambda container...${NC}"
    
    # Login a ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Crear repositorio ECR para Lambda
    LAMBDA_REPO="$ECR_REPOSITORY-lambda"
    if ! aws ecr describe-repositories --repository-names $LAMBDA_REPO --region $AWS_REGION &> /dev/null; then
        echo "Creating ECR repository for Lambda..."
        aws ecr create-repository --repository-name $LAMBDA_REPO --region $AWS_REGION
    fi
    
    # Build y push
    ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$LAMBDA_REPO:$IMAGE_TAG"
    
    echo "Building Lambda container image..."
    docker build -t $LAMBDA_REPO:$IMAGE_TAG -f Dockerfile.lambda .
    
    echo "Tagging image..."
    docker tag $LAMBDA_REPO:$IMAGE_TAG $ECR_URI
    
    echo "Pushing image to ECR..."
    docker push $ECR_URI
    
    # Actualizar función Lambda
    echo "Updating Lambda function..."
    aws lambda update-function-code \
        --function-name $LAMBDA_FUNCTION_NAME \
        --image-uri $ECR_URI \
        --region $AWS_REGION
    
    echo -e "${GREEN}✅ Lambda container deployment completed!${NC}"
}

# Menú principal
case "${1:-ecs}" in
    ecs)
        deploy_ecs
        ;;
    lambda)
        deploy_lambda
        ;;
    lambda-container)
        deploy_lambda_container
        ;;
    *)
        echo "Usage: $0 [ecs|lambda|lambda-container]"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"




