# Deployment Guide

This guide covers deployment options for the Logistics AI Platform.

## 🐳 Docker Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

### Quick Start

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Environment Variables

Create a `.env` file or set environment variables:

```env
# Server
HOST=0.0.0.0
PORT=8030
LOG_LEVEL=INFO

# Redis
REDIS_URL=redis://redis:6379/0

# Database
DATABASE_URL=sqlite+aiosqlite:///./logistics.db

# Security
SECRET_KEY=your-secret-key-change-in-production

# External APIs (optional)
GOOGLE_MAPS_API_KEY=your_key
WEATHER_API_KEY=your_key
OPENAI_API_KEY=your_key
```

### Docker Build

```bash
# Build image
docker build -t logistics-ai-platform:latest .

# Run container
docker run -d \
  --name logistics-api \
  -p 8030:8030 \
  -e REDIS_URL=redis://your-redis:6379/0 \
  logistics-ai-platform:latest
```

## ☁️ Production Deployment

### Health Checks

The application includes health check endpoints:

- `/health` - Service health status
- `/ready` - Readiness check for orchestration

### Kubernetes

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: logistics-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: logistics-api
  template:
    metadata:
      labels:
        app: logistics-api
    spec:
      containers:
      - name: api
        image: logistics-ai-platform:latest
        ports:
        - containerPort: 8030
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: logistics-secrets
              key: redis-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8030
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8030
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Monitoring

The application exposes Prometheus metrics at `/metrics`:

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'logistics-api'
    static_configs:
      - targets: ['logistics-api:8030']
    metrics_path: '/metrics'
```

## 🔧 Configuration

### Logging

Logs are written to:
- Console (stdout)
- `logs/app_YYYY-MM-DD.log` (all logs)
- `logs/error_YYYY-MM-DD.log` (errors only)

### Performance Tuning

For production, consider:

1. **Uvicorn Workers**: Use multiple workers for better concurrency
   ```bash
   uvicorn main:app --workers 4 --host 0.0.0.0 --port 8030
   ```

2. **Redis**: Configure Redis for distributed caching
   ```env
   REDIS_URL=redis://redis-cluster:6379/0
   ```

3. **Database**: Use PostgreSQL for production
   ```env
   DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/logistics
   ```

## 📊 Monitoring

### Health Checks

```bash
# Check health
curl http://localhost:8030/health

# Check readiness
curl http://localhost:8030/ready
```

### Metrics

```bash
# Get Prometheus metrics
curl http://localhost:8030/metrics

# Get metrics info
curl http://localhost:8030/metrics/info
```

## 🚀 CI/CD

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t logistics-api:${{ github.sha }} .
      - name: Deploy
        run: |
          # Your deployment commands
```

## 🔒 Security

### Production Checklist

- [ ] Change `SECRET_KEY` to a secure random value
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Enable authentication/authorization
- [ ] Use environment variables for secrets
- [ ] Enable security headers
- [ ] Regular dependency updates
- [ ] Enable logging and monitoring
- [ ] Set up backup strategy

## 📝 Environment-Specific Configs

### Development

```env
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

### Production

```env
LOG_LEVEL=INFO
ENVIRONMENT=production
REDIS_URL=redis://production-redis:6379/0
DATABASE_URL=postgresql+asyncpg://...
```

