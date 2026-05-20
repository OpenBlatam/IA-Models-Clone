# Deployment Guide - Dermatology AI

## Overview

This guide covers deployment strategies for the Dermatology AI service across different environments and platforms.

## Deployment Options

### 1. Docker Deployment

#### Prerequisites
- Docker
- Docker Compose

#### Quick Start
```bash
# Build and start services
docker-compose up -d

# Check health
python3 scripts/health_check.py

# View logs
docker-compose logs -f dermatology-ai
```

#### Production Deployment
```bash
# Build production image
docker build -t dermatology-ai:6.1.0 .

# Run with production settings
docker run -d \
  --name dermatology-ai \
  -p 8006:8006 \
  -e ENVIRONMENT=production \
  -e REDIS_URL=redis://redis:6379/0 \
  -e JWT_SECRET=your-secret-key \
  dermatology-ai:6.1.0
```

### 2. Kubernetes Deployment

#### Prerequisites
- kubectl
- Kubernetes cluster
- Helm (optional)

#### Deploy with kubectl
```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=dermatology-ai

# Check service
kubectl get svc dermatology-ai
```

#### Deploy with Helm
```bash
# Install Helm chart
helm install dermatology-ai ./helm/dermatology-ai

# Upgrade
helm upgrade dermatology-ai ./helm/dermatology-ai
```

### 3. AWS Lambda (Serverless)

#### Prerequisites
- AWS CLI configured
- Serverless Framework
- AWS credentials

#### Deploy
```bash
# Install serverless
npm install -g serverless

# Deploy
serverless deploy --stage production

# Or use script
./scripts/deploy.sh lambda
```

#### Configuration
```yaml
# serverless.yml
service: dermatology-ai

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  memorySize: 512
  timeout: 30

functions:
  api:
    handler: handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
```

### 4. Azure Functions

#### Prerequisites
- Azure CLI
- Azure Functions Core Tools

#### Deploy
```bash
# Login to Azure
az login

# Create function app
az functionapp create \
  --resource-group myResourceGroup \
  --consumption-plan-location eastus \
  --runtime python \
  --functions-version 4 \
  --name dermatology-ai \
  --storage-account mystorageaccount

# Deploy
func azure functionapp publish dermatology-ai
```

### 5. Google Cloud Run

#### Prerequisites
- gcloud CLI
- Docker

#### Deploy
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/dermatology-ai

# Deploy to Cloud Run
gcloud run deploy dermatology-ai \
  --image gcr.io/PROJECT_ID/dermatology-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Environment Configuration

### Required Variables
```bash
# Core
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security
JWT_SECRET=your-secret-key-here
REQUIRE_AUTH=true

# Database
DATABASE_TYPE=dynamodb  # or cosmosdb, sqlite, postgresql
COSMOSDB_ENDPOINT=https://...
COSMOSDB_KEY=...

# Cache
REDIS_URL=redis://redis:6379/0
USE_REDIS=true

# Message Broker
MESSAGE_BROKER_TYPE=rabbitmq  # or kafka, memory
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

# Observability
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
```

### Optional Variables
```bash
# API Gateway
API_GATEWAY_TYPE=kong
API_GATEWAY_URL=http://kong:8001

# Service Discovery
SERVICE_DISCOVERY_TYPE=consul
CONSUL_URL=http://consul:8500

# Service Mesh
SERVICE_MESH_TYPE=istio
KUBERNETES_NAMESPACE=default

# Elasticsearch
ELASTICSEARCH_HOSTS=elasticsearch:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=...
```

## Health Checks

### Basic Health Check
```bash
curl http://localhost:8006/health
```

### Detailed Health Check
```bash
curl http://localhost:8006/dermatology/health/detailed
```

### Using Script
```bash
# Basic
python3 scripts/health_check.py

# Detailed
python3 scripts/health_check.py --detailed

# Readiness probe
python3 scripts/health_check.py --readiness

# Liveness probe
python3 scripts/health_check.py --liveness
```

## Monitoring

### Prometheus Metrics
```bash
# Scrape metrics
curl http://localhost:8006/metrics
```

### Grafana Dashboard
1. Import dashboard from `monitoring/grafana-dashboard.json`
2. Configure Prometheus as data source
3. View metrics and alerts

### Logging
```bash
# View logs (Docker)
docker-compose logs -f dermatology-ai

# View logs (Kubernetes)
kubectl logs -f deployment/dermatology-ai

# View logs (Cloud)
# Use cloud provider's logging service
```

## Scaling

### Horizontal Scaling (Kubernetes)
```bash
# Scale deployment
kubectl scale deployment dermatology-ai --replicas=5

# Auto-scaling
kubectl autoscale deployment dermatology-ai \
  --min=2 --max=10 --cpu-percent=70
```

### Vertical Scaling
```yaml
# Update resources in deployment
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## Security

### Secrets Management

#### Kubernetes
```bash
# Create secret
kubectl create secret generic dermatology-secrets \
  --from-literal=jwt-secret=your-secret \
  --from-literal=db-password=your-password
```

#### Docker
```bash
# Use Docker secrets or environment files
docker run --env-file .env.production dermatology-ai
```

### Network Security
- Use service mesh (Istio/Linkerd) for mTLS
- Configure network policies
- Use API Gateway for rate limiting
- Enable DDoS protection

## Backup & Recovery

### Database Backup
```bash
# DynamoDB
aws dynamodb create-backup --table-name dermatology-analyses

# Cosmos DB
az cosmosdb sql container backup \
  --account-name myaccount \
  --database-name mydb \
  --name mycontainer
```

### Configuration Backup
```bash
# Export configuration
kubectl get configmap dermatology-config -o yaml > config-backup.yaml
```

## Troubleshooting

### Common Issues

#### Service Not Starting
```bash
# Check logs
docker-compose logs dermatology-ai

# Check health
python3 scripts/health_check.py --detailed
```

#### High Memory Usage
```bash
# Check memory
docker stats dermatology-ai

# Profile application
py-spy record -o profile.svg -- python main.py
```

#### Connection Issues
```bash
# Test Redis connection
redis-cli -h redis ping

# Test database connection
python3 -c "from utils.database_abstraction import get_database_adapter; ..."
```

## Performance Tuning

### Cold Start Optimization
- Use `requirements-minimal.txt` for serverless
- Enable connection pooling
- Use lazy loading for heavy dependencies
- Optimize Docker image size

### Throughput Optimization
- Enable multiple workers (uvicorn)
- Use Redis for distributed caching
- Implement connection pooling
- Use async/await throughout

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy
        run: ./scripts/deploy.sh kubernetes
```

### GitLab CI
```yaml
# .gitlab-ci.yml
deploy:
  stage: deploy
  script:
    - ./scripts/deploy.sh kubernetes
```

## Best Practices

1. **Always use health checks** before routing traffic
2. **Monitor metrics** continuously
3. **Use secrets management** for sensitive data
4. **Enable auto-scaling** for variable loads
5. **Implement circuit breakers** for resilience
6. **Use structured logging** for observability
7. **Test deployments** in staging first
8. **Keep dependencies updated** for security
9. **Use multi-stage Docker builds** for smaller images
10. **Implement blue-green deployments** for zero downtime

## Support

For issues or questions:
- Check logs first
- Review health check output
- Consult monitoring dashboards
- Review deployment configuration















