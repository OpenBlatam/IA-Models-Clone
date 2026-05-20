# Quick Start Guide - Cloud Deployment

## AWS Lambda (Serverless) - 5 Minutes

### Prerequisites
- AWS CLI configured
- AWS credentials with Lambda, API Gateway, and IAM permissions

### Steps

1. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r deployment/requirements-serverless.txt
```

2. **Deploy with Terraform:**
```bash
cd deployment/aws/terraform
terraform init
terraform apply \
  -var="environment=production" \
  -var="spotify_client_id=YOUR_CLIENT_ID" \
  -var="spotify_client_secret=YOUR_CLIENT_SECRET"
```

3. **Get API URL:**
```bash
terraform output api_url
```

That's it! Your API is live on AWS Lambda.

## Azure Functions (Serverless) - 5 Minutes

### Prerequisites
- Azure CLI installed
- Azure subscription
- Azure Functions Core Tools v4

### Steps

1. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r deployment/requirements-serverless.txt
npm install -g azure-functions-core-tools@4
```

2. **Deploy with ARM template:**
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

3. **Deploy function code:**
```bash
func azure functionapp publish music-analyzer-ai-prod --python
```

Your API is now live on Azure Functions!

## Docker (Any Cloud) - 3 Minutes

### Steps

1. **Build image:**
```bash
docker build -f deployment/Dockerfile -t music-analyzer-ai:latest .
```

2. **Run locally:**
```bash
docker run -p 8010:8010 \
  -e SPOTIFY_CLIENT_ID=YOUR_CLIENT_ID \
  -e SPOTIFY_CLIENT_SECRET=YOUR_CLIENT_SECRET \
  music-analyzer-ai:latest
```

3. **Deploy to cloud:**
   - **AWS ECS:** Use `deployment/aws/ecs-task-definition.json`
   - **Azure Container Instances:** Use `deployment/azure/container-instance.yaml`
   - **Google Cloud Run:** `gcloud run deploy --source .`

## Testing Your Deployment

### Health Check
```bash
curl https://your-api-url/health
```

### Test Endpoint
```bash
curl -X POST https://your-api-url/music/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Bohemian Rhapsody", "limit": 5}'
```

## Next Steps

1. **Configure monitoring:**
   - AWS: Set up CloudWatch dashboards
   - Azure: Configure Application Insights

2. **Set up CI/CD:**
   - Use GitHub Actions workflows in `.github/workflows/`

3. **Secure your API:**
   - Add API keys or OAuth2
   - Configure rate limiting
   - Use secrets management

4. **Optimize performance:**
   - Enable caching
   - Configure connection pooling
   - Monitor cold starts

## Troubleshooting

### AWS Lambda
- **Cold starts:** Increase memory, use provisioned concurrency
- **Timeouts:** Increase timeout in Lambda configuration
- **Import errors:** Check Lambda Layer includes all dependencies

### Azure Functions
- **Cold starts:** Use Premium plan or enable pre-warming
- **Timeouts:** Increase `functionTimeout` in `host.json`
- **Import errors:** Ensure all packages are in `requirements.txt`

### Docker
- **Build fails:** Check Dockerfile and dependencies
- **Runtime errors:** Check environment variables
- **Port issues:** Verify port mapping

## Support

For detailed documentation, see `deployment/README.md`




