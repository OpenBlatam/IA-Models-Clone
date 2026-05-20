#!/bin/bash
# Deploy ECS service

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
CLUSTER_NAME=${CLUSTER_NAME:-faceless-video-ai-production}
SERVICE_NAME=${SERVICE_NAME:-faceless-video-ai-service}
TASK_DEFINITION_FILE="ecs-task-definition.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deploying ECS Service${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Cluster: ${CLUSTER_NAME}"
echo -e "Service: ${SERVICE_NAME}"
echo -e "Region: ${AWS_REGION}"
echo ""

# Check if task definition file exists
if [ ! -f "$TASK_DEFINITION_FILE" ]; then
    echo -e "${RED}Error: Task definition file not found: ${TASK_DEFINITION_FILE}${NC}"
    exit 1
fi

# Register new task definition
echo -e "${YELLOW}Registering new task definition...${NC}"
TASK_DEF_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://${TASK_DEFINITION_FILE} \
    --region ${AWS_REGION} \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo -e "${GREEN}Task definition registered: ${TASK_DEF_ARN}${NC}"

# Check if service exists
SERVICE_EXISTS=$(aws ecs describe-services \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'services[0].status' \
    --output text 2>/dev/null || echo "MISSING")

if [ "$SERVICE_EXISTS" = "MISSING" ] || [ "$SERVICE_EXISTS" = "None" ]; then
    echo -e "${YELLOW}Service does not exist. Creating new service...${NC}"
    
    # Create service using service definition
    if [ -f "ecs-service-definition.json" ]; then
        # Update task definition in service definition
        sed "s|\"taskDefinition\": \".*\"|\"taskDefinition\": \"${TASK_DEF_ARN}\"|g" \
            ecs-service-definition.json > /tmp/service-def.json
        
        aws ecs create-service \
            --cli-input-json file:///tmp/service-def.json \
            --region ${AWS_REGION}
        
        echo -e "${GREEN}Service created successfully${NC}"
    else
        echo -e "${RED}Error: Service definition file not found${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Service exists. Updating service...${NC}"
    
    # Update service with new task definition
    aws ecs update-service \
        --cluster ${CLUSTER_NAME} \
        --service ${SERVICE_NAME} \
        --task-definition ${TASK_DEF_ARN} \
        --force-new-deployment \
        --region ${AWS_REGION} \
        --query 'service.[serviceName,status,desiredCount,runningCount]' \
        --output table
    
    echo -e "${GREEN}Service update initiated${NC}"
fi

# Wait for service to stabilize
echo -e "${YELLOW}Waiting for service to stabilize...${NC}"
aws ecs wait services-stable \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment completed successfully${NC}"
echo -e "${GREEN}========================================${NC}"

# Display service status
echo -e "${YELLOW}Service status:${NC}"
aws ecs describe-services \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'services[0].[serviceName,status,desiredCount,runningCount,pendingCount]' \
    --output table




