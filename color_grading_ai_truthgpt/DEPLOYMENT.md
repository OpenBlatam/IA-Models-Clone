# Guía de Deployment - Color Grading AI TruthGPT

## Deployment Options

### 1. Docker Deployment

#### Build Image

```bash
docker build -t color-grading-ai -f docker/Dockerfile .
```

#### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY="your-key" \
  -v $(pwd)/color_grading_output:/app/color_grading_output \
  color-grading-ai
```

#### Docker Compose

```bash
cd docker
docker-compose up -d
```

### 2. Kubernetes Deployment

#### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: color-grading-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: color-grading-ai
  template:
    metadata:
      labels:
        app: color-grading-ai
    spec:
      containers:
      - name: api
        image: color-grading-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: color-grading-secrets
              key: openrouter-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### 3. Cloud Deployment

#### AWS ECS/Fargate

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t color-grading-ai .
docker tag color-grading-ai:latest <account>.dkr.ecr.us-east-1.amazonaws.com/color-grading-ai:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/color-grading-ai:latest
```

#### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/<project>/color-grading-ai
gcloud run deploy color-grading-ai \
  --image gcr.io/<project>/color-grading-ai \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENROUTER_API_KEY=<key>
```

## Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your-api-key

# Optional
TRUTHGPT_ENDPOINT=https://your-truthgpt-endpoint
FFMPEG_PATH=/usr/bin/ffmpeg
LOG_LEVEL=INFO
MAX_PARALLEL_TASKS=5
CACHE_TTL=3600
```

## Production Checklist

- [ ] Configure API keys and secrets
- [ ] Set up health check monitoring
- [ ] Configure rate limiting
- [ ] Set up logging aggregation
- [ ] Configure backup strategy
- [ ] Set up cloud storage integration
- [ ] Configure autoscaling
- [ ] Set up monitoring and alerts
- [ ] Configure SSL/TLS
- [ ] Set up CI/CD pipeline

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Metrics

```bash
# Get metrics
curl http://localhost:8000/api/v1/metrics

# Dashboard stats
curl http://localhost:8000/api/v1/dashboard/stats
```

## Scaling

### Horizontal Scaling

- Use load balancer
- Configure multiple replicas
- Use shared storage for cache
- Use distributed queue (Redis/RabbitMQ)

### Vertical Scaling

- Increase CPU/memory limits
- Optimize FFmpeg settings
- Use GPU acceleration (if available)

## Security

- Use API keys for authentication
- Enable HTTPS/TLS
- Configure CORS appropriately
- Set up rate limiting
- Regular security updates
- Monitor for abuse

## Backup Strategy

- Regular backups of:
  - Presets
  - Templates
  - History
  - Configuration

```bash
# Create backup
curl -X POST http://localhost:8000/api/v1/backup/create \
  -d '{"source_dirs": ["presets", "templates"]}'
```




