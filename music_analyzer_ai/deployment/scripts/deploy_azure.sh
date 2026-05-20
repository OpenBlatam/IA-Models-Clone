#!/bin/bash
# Deployment script for Azure Functions/Container Instances

set -e

ENVIRONMENT=${1:-production}
RESOURCE_GROUP=${2:-music-analyzer-rg}
LOCATION=${3:-eastus}

echo "Deploying Music Analyzer AI to Azure (Environment: $ENVIRONMENT, Resource Group: $RESOURCE_GROUP)"

# Check if using Functions or Container Instances
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-functions}

if [ "$DEPLOYMENT_TYPE" == "functions" ]; then
    echo "Deploying to Azure Functions..."
    
    FUNCTION_APP_NAME="music-analyzer-ai-$ENVIRONMENT"
    
    # Create resource group if it doesn't exist
    az group create --name $RESOURCE_GROUP --location $LOCATION || true
    
    # Deploy using ARM template
    echo "Deploying ARM template..."
    az deployment group create \
        --resource-group $RESOURCE_GROUP \
        --template-file deployment/azure/arm_template.json \
        --parameters \
            environment=$ENVIRONMENT \
            functionAppName=$FUNCTION_APP_NAME \
            storageAccountName="musicanalyzer${ENVIRONMENT}sa" \
            appServicePlanName="music-analyzer-plan-$ENVIRONMENT" \
            spotifyClientId=$SPOTIFY_CLIENT_ID \
            spotifyClientSecret=$SPOTIFY_CLIENT_SECRET \
            location=$LOCATION
    
    # Deploy function code
    echo "Deploying function code..."
    cd ..
    func azure functionapp publish $FUNCTION_APP_NAME --python
    
elif [ "$DEPLOYMENT_TYPE" == "container" ]; then
    echo "Deploying to Azure Container Instances..."
    
    # Build Docker image
    echo "Building Docker image..."
    docker build -f deployment/Dockerfile -t music-analyzer-ai:latest .
    
    # Create Azure Container Registry if needed
    ACR_NAME="musicanalyzerai"
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic || true
    
    # Login to ACR
    az acr login --name $ACR_NAME
    
    # Tag and push image
    docker tag music-analyzer-ai:latest $ACR_NAME.azurecr.io/music-analyzer-ai:latest
    docker push $ACR_NAME.azurecr.io/music-analyzer-ai:latest
    
    # Deploy container instance
    az container create \
        --resource-group $RESOURCE_GROUP \
        --name music-analyzer-ai \
        --image $ACR_NAME.azurecr.io/music-analyzer-ai:latest \
        --registry-login-server $ACR_NAME.azurecr.io \
        --ip-address Public \
        --ports 8010 \
        --environment-variables \
            ENVIRONMENT=$ENVIRONMENT \
            CACHE_ENABLED=true \
            LOG_LEVEL=INFO \
        --secure-environment-variables \
            SPOTIFY_CLIENT_ID=$SPOTIFY_CLIENT_ID \
            SPOTIFY_CLIENT_SECRET=$SPOTIFY_CLIENT_SECRET
fi

echo "Deployment complete!"




