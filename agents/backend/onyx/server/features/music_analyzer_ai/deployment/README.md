# Cloud Deployment Guide - Music Analyzer AI

This guide covers deploying Music Analyzer AI to AWS and Azure using serverless and containerized architectures.

## 🚀 Quick Start - One Command

### Local Development

**Start everything with a single command:**

```bash
# Windows
deployment\start.bat

# Linux/Mac
./deployment/start.sh

# Python (cross-platform)
python deployment/start.py

# Make (Linux/Mac)
make start
```

**Stop everything:**

```bash
# Windows
deployment\stop.bat

# Linux/Mac
./deployment/stop.sh

# Make
make stop
```

### EC2 Deployment (Any Instance)

**Deploy on any EC2 instance:**

```bash
# SSH to your EC2 instance, then:
curl -fsSL https://raw.githubusercontent.com/your-repo/music-analyzer-ai/main/deployment/ec2/deploy.sh | bash
```

Or use the quick setup:

```bash
./deployment/ec2/quick-deploy.sh
```

See `README_START.md` for local development guide.
See `ec2/README.md` for EC2 deployment guide.

## Table of Contents

- [AWS Deployment](#aws-deployment)
  - [Lambda (Serverless)](#lambda-serverless)
  - [ECS/Fargate (Containers)](#ecsfargate-containers)
- [Azure Deployment](#azure-deployment)
  - [Azure Functions (Serverless)](#azure-functions-serverless)
  - [Container Instances (Containers)](#container-instances-containers)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## AWS Deployment

### Lambda (Serverless)

#### Prerequisites

- AWS CLI configured
- Terraform (optional) or AWS CloudFormation
- Python 3.11
- Docker (for building layers)

#### Quick Start

1. **Install serverless dependencies:**
```bash
pip install -r deployment/requirements-serverless.txt
```

2. **Deploy using Terraform:**
```bash
cd deployment/aws/terraform
terraform init
terraform plan -var="environment=production" \
               -var="spotify_client_id=YOUR_CLIENT_ID" \
               -var="spotify_client_secret=YOUR_CLIENT_SECRET"
terraform apply
```

3. **Deploy using CloudFormation:**
```bash
aws cloudformation create-stack \
  --stack-name music-analyzer-ai \
  --template-body file://deployment/aws/cloudformation.yaml \
  --parameters ParameterKey=Environment,ParameterValue=production \
               ParameterKey=SpotifyClientId,ParameterValue=YOUR_CLIENT_ID \
               ParameterKey=SpotifyClientSecret,ParameterValue=YOUR_CLIENT_SECRET
```

4. **Deploy using script:**
```bash
chmod +x deployment/scripts/deploy_aws.sh
DEPLOYMENT_TYPE=lambda ./deployment/scripts/deploy_aws.sh production us-east-1
```

#### Configuration

Set environment variables in Lambda:
- `SPOTIFY_CLIENT_ID`: Spotify API Client ID
- `SPOTIFY_CLIENT_SECRET`: Spotify API Client Secret
- `ENVIRONMENT`: production/staging/development
- `CACHE_ENABLED`: true/false
- `LOG_LEVEL`: INFO/DEBUG/WARNING

#### API Gateway

The deployment automatically creates an HTTP API Gateway. The endpoint URL will be provided in the Terraform/CloudFormation outputs.

### ECS/Fargate (Containers)

#### Prerequisites

- AWS CLI configured
- Docker
- ECR repository created

#### Quick Start

1. **Build and push Docker image:**
```bash
docker build -f deployment/Dockerfile -t music-analyzer-ai:latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag music-analyzer-ai:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/music-analyzer-ai:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/music-analyzer-ai:latest
```

2. **Deploy using script:**
```bash
DEPLOYMENT_TYPE=ecs ./deployment/scripts/deploy_aws.sh production us-east-1
```

3. **Update ECS task definition:**
```bash
aws ecs register-task-definition --cli-input-json file://deployment/aws/ecs-task-definition.json
```

## Azure Deployment

### Azure Functions (Serverless)

#### Prerequisites

- Azure CLI installed and configured
- Azure Functions Core Tools v4
- Python 3.11

#### Quick Start

1. **Install Azure Functions Core Tools:**
```bash
npm install -g azure-functions-core-tools@4 --unsafe-perm true
```

2. **Install serverless dependencies:**
```bash
pip install -r deployment/requirements-serverless.txt
```

3. **Deploy using ARM template:**
```bash
az deployment group create \
  --resource-group music-analyzer-rg \
  --template-file deployment/azure/arm_template.json \
  --parameters \
    environment=production \
    functionAppName=music-analyzer-ai-prod \
    storageAccountName=musicanalyzerprodsa \
    appServicePlanName=music-analyzer-plan-prod \
    spotifyClientId=YOUR_CLIENT_ID \
    spotifyClientSecret=YOUR_CLIENT_SECRET
```

4. **Deploy function code:**
```bash
cd ..
func azure functionapp publish music-analyzer-ai-prod --python
```

5. **Deploy using script:**
```bash
chmod +x deployment/scripts/deploy_azure.sh
DEPLOYMENT_TYPE=functions \
SPOTIFY_CLIENT_ID=YOUR_CLIENT_ID \
SPOTIFY_CLIENT_SECRET=YOUR_CLIENT_SECRET \
./deployment/scripts/deploy_azure.sh production music-analyzer-rg eastus
```

### Container Instances (Containers)

#### Prerequisites

- Azure CLI configured
- Docker
- Azure Container Registry

#### Quick Start

1. **Create Azure Container Registry:**
```bash
az acr create --resource-group music-analyzer-rg --name musicanalyzerai --sku Basic
```

2. **Build and push image:**
```bash
docker build -f deployment/Dockerfile -t music-analyzer-ai:latest .
az acr login --name musicanalyzerai
docker tag music-analyzer-ai:latest musicanalyzerai.azurecr.io/music-analyzer-ai:latest
docker push musicanalyzerai.azurecr.io/music-analyzer-ai:latest
```

3. **Deploy container instance:**
```bash
az container create \
  --resource-group music-analyzer-rg \
  --name music-analyzer-ai \
  --image musicanalyzerai.azurecr.io/music-analyzer-ai:latest \
  --registry-login-server musicanalyzerai.azurecr.io \
  --ip-address Public \
  --ports 8010 \
  --environment-variables ENVIRONMENT=production CACHE_ENABLED=true \
  --secure-environment-variables SPOTIFY_CLIENT_ID=YOUR_CLIENT_ID SPOTIFY_CLIENT_SECRET=YOUR_CLIENT_SECRET
```

4. **Deploy using script:**
```bash
DEPLOYMENT_TYPE=container \
SPOTIFY_CLIENT_ID=YOUR_CLIENT_ID \
SPOTIFY_CLIENT_SECRET=YOUR_CLIENT_SECRET \
./deployment/scripts/deploy_azure.sh production music-analyzer-rg eastus
```

## Configuration

### Environment Variables

Both AWS and Azure deployments support the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name | `production` |
| `SPOTIFY_CLIENT_ID` | Spotify API Client ID | Required |
| `SPOTIFY_CLIENT_SECRET` | Spotify API Client Secret | Required |
| `CACHE_ENABLED` | Enable caching | `true` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8010` |

### Secrets Management

#### AWS

Use AWS Secrets Manager or Systems Manager Parameter Store:

```bash
aws secretsmanager create-secret \
  --name music-analyzer/spotify-client-id \
  --secret-string YOUR_CLIENT_ID

aws secretsmanager create-secret \
  --name music-analyzer/spotify-client-secret \
  --secret-string YOUR_CLIENT_SECRET
```

#### Azure

Use Azure Key Vault:

```bash
az keyvault secret set --vault-name music-analyzer-kv --name spotify-client-id --value YOUR_CLIENT_ID
az keyvault secret set --vault-name music-analyzer-kv --name spotify-client-secret --value YOUR_CLIENT_SECRET
```

## Monitoring

### AWS CloudWatch

1. **View logs:**
```bash
aws logs tail /aws/lambda/music-analyzer-ai-production --follow
```

2. **Create dashboard:**
```bash
aws cloudwatch put-dashboard \
  --dashboard-name music-analyzer-dashboard \
  --dashboard-body file://deployment/monitoring/cloudwatch_dashboard.json
```

3. **Set up alarms:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name music-analyzer-errors \
  --alarm-description "Alert on Lambda errors" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold
```

### Azure Application Insights

1. **View metrics in Azure Portal:**
   - Navigate to your Function App
   - Open Application Insights
   - View pre-configured queries

2. **Custom queries:**
   - Use KQL queries from `deployment/monitoring/application_insights.json`

3. **Set up alerts:**
   - Navigate to Application Insights > Alerts
   - Create alert rules based on metrics

## Performance Optimization

### Cold Start Reduction

1. **Use Lambda Layers:**
   - Package dependencies in a Lambda Layer
   - Reuse across multiple functions

2. **Lazy Loading:**
   - Already implemented in handlers
   - Services load on first request

3. **Connection Pooling:**
   - Reuse HTTP connections
   - Configure in `serverless_config.py`

### Scaling

#### AWS Lambda
- Automatic scaling up to account limits
- Configure reserved concurrency if needed
- Use provisioned concurrency for critical functions

#### Azure Functions
- Consumption plan: automatic scaling
- Premium plan: pre-warmed instances
- Dedicated plan: fixed capacity

## Troubleshooting

### Common Issues

1. **Cold Start Timeout:**
   - Increase Lambda/Functions timeout
   - Use provisioned concurrency (AWS) or Premium plan (Azure)

2. **Memory Issues:**
   - Increase Lambda memory (affects CPU allocation)
   - Monitor memory usage in CloudWatch/Application Insights

3. **API Gateway Timeout:**
   - Default timeout is 30 seconds
   - Use async processing for long-running tasks

4. **Import Errors:**
   - Ensure all dependencies are in deployment package
   - Check Lambda Layer includes all packages

### Debugging

#### AWS
```bash
# View recent logs
aws logs tail /aws/lambda/music-analyzer-ai-production --since 1h

# Test Lambda locally
sam local start-api
```

#### Azure
```bash
# View logs
az functionapp log tail --name music-analyzer-ai-prod --resource-group music-analyzer-rg

# Test locally
func start
```

## Security Best Practices

1. **Secrets Management:**
   - Never commit secrets to code
   - Use AWS Secrets Manager or Azure Key Vault

2. **Network Security:**
   - Use VPC for Lambda (if needed)
   - Configure security groups for ECS

3. **API Security:**
   - Enable API Gateway authentication
   - Use API keys or OAuth2
   - Implement rate limiting

4. **Container Security:**
   - Scan images for vulnerabilities
   - Use least privilege IAM roles
   - Enable encryption at rest

## Cost Optimization

### AWS
- Use Lambda for variable workloads
- Use ECS/Fargate for consistent workloads
- Enable Lambda provisioned concurrency only when needed
- Use DynamoDB on-demand billing

### Azure
- Use Consumption plan for variable workloads
- Use Premium plan for consistent workloads
- Enable auto-shutdown for dev/test environments
- Use Azure Container Instances for burst workloads

## Support

For issues or questions:
- Check logs in CloudWatch/Application Insights
- Review deployment scripts for errors
- Consult AWS/Azure documentation
- Open an issue in the repository

