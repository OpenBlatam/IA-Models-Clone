# 🚀 Deployment Guide - Improved Video-OpusClip API

## 📋 **Production Deployment Checklist**

This guide provides step-by-step instructions for deploying the improved Video-OpusClip API to production environments.

---

## 🏗️ **Architecture Overview**

### **Production Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   FastAPI App   │────│   Redis Cache   │
│   (Nginx/HAProxy)│    │   (Multiple)    │    │   (Primary)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Database      │    │   Monitoring    │
                       │   (PostgreSQL)  │    │   (Prometheus)  │
                       └─────────────────┘    └─────────────────┘
```

### **Key Components**
- **FastAPI Application**: Main API server with improved architecture
- **Redis Cache**: High-performance caching layer
- **PostgreSQL**: Persistent data storage
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection and monitoring
- **Docker**: Containerization for easy deployment

---

## 🐳 **Docker Deployment**

### **1. Dockerfile**

```dockerfile
# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_opus_clip.txt .
RUN pip install --no-cache-dir -r requirements_opus_clip.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "improved_api.py"]
```

### **2. Docker Compose**

```yaml
version: '3.8'

services:
  # FastAPI Application
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_DB=0
      - DATABASE_URL=postgresql://user:password@postgres:5432/video_api
      - LOG_LEVEL=info
      - WORKERS=4
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=video_api
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
```

### **3. Nginx Configuration**

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
        # Add more servers for load balancing
        # server api2:8000;
        # server api3:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Rate Limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
        limit_req zone=api burst=20 nodelay;

        # API Routes
        location /api/ {
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Health Check
        location /health {
            proxy_pass http://api_backend;
            access_log off;
        }

        # Static Files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

---

## ☁️ **Cloud Deployment**

### **AWS Deployment**

#### **1. ECS with Fargate**

```yaml
# task-definition.json
{
  "family": "video-opusclip-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/video-opusclip-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "REDIS_HOST",
          "value": "your-redis-cluster.cache.amazonaws.com"
        },
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:password@your-rds-endpoint:5432/video_api"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/video-opusclip-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### **2. Application Load Balancer**

```yaml
# alb.yaml
apiVersion: v1
kind: Service
metadata:
  name: video-opusclip-alb
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: arn:aws:acm:region:account:certificate/cert-id
    service.beta.kubernetes.io/aws-load-balancer-ssl-ports: https
spec:
  type: LoadBalancer
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: https
    port: 443
    targetPort: 8000
  selector:
    app: video-opusclip-api
```

### **Google Cloud Platform**

#### **1. Cloud Run**

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: video-opusclip-api
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/your-project/video-opusclip-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "your-redis-instance"
        - name: DATABASE_URL
          value: "postgresql://user:password@your-cloud-sql:5432/video_api"
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
```

### **Azure**

#### **1. Container Instances**

```yaml
# azure-container-instance.yaml
apiVersion: 2018-10-01
location: eastus
name: video-opusclip-api
properties:
  containers:
  - name: api
    properties:
      image: your-registry.azurecr.io/video-opusclip-api:latest
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: REDIS_HOST
        value: your-redis-cache.redis.cache.windows.net
      - name: DATABASE_URL
        value: postgresql://user:password@your-postgres-server.postgres.database.azure.com:5432/video_api
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
```

---

## 🔧 **Environment Configuration**

### **Environment Variables**

```bash
# Application Configuration
APP_NAME=video-opusclip-api
APP_VERSION=2.0.0
LOG_LEVEL=info
WORKERS=4
HOST=0.0.0.0
PORT=8000

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/video_api
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
PROMETHEUS_ENDPOINT=/metrics

# Performance Configuration
MAX_WORKERS=8
BATCH_SIZE=10
CACHE_TTL=3600
REQUEST_TIMEOUT=300

# External Services
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### **Production Configuration File**

```python
# config/production.py
import os
from typing import Optional

class ProductionConfig:
    """Production configuration for Video-OpusClip API."""
    
    # Application
    APP_NAME: str = os.getenv("APP_NAME", "video-opusclip-api")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    WORKERS: int = int(os.getenv("WORKERS", "4"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/video_api")
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "30"))
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    PROMETHEUS_ENDPOINT: str = os.getenv("PROMETHEUS_ENDPOINT", "/metrics")
    
    # Performance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "8"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
    
    # External Services
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
```

---

## 📊 **Monitoring & Observability**

### **Prometheus Configuration**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'video-opusclip-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### **Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "Video-OpusClip API Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])",
            "legendFormat": "Hit Rate"
          }
        ]
      }
    ]
  }
}
```

---

## 🔒 **Security Best Practices**

### **1. SSL/TLS Configuration**

```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Use Let's Encrypt for production
certbot --nginx -d your-domain.com
```

### **2. Security Headers**

```python
# security_headers.py
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware

def add_security_headers(app: FastAPI):
    """Add security headers to FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://your-domain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["your-domain.com", "*.your-domain.com"]
    )
```

### **3. Rate Limiting**

```python
# rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply rate limiting to endpoints
@app.post("/api/v1/video/process")
@limiter.limit("10/minute")
async def process_video(request: Request, video_request: VideoClipRequest):
    # Processing logic
    pass
```

---

## 🚀 **Deployment Commands**

### **Docker Deployment**

```bash
# Build and deploy
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Scale services
docker-compose up -d --scale api=3
```

### **Kubernetes Deployment**

```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment video-opusclip-api --replicas=5

# Check logs
kubectl logs -f deployment/video-opusclip-api
```

### **Cloud Deployment**

```bash
# AWS ECS
aws ecs create-service --cluster your-cluster --service-name video-opusclip-api --task-definition video-opusclip-api

# Google Cloud Run
gcloud run deploy video-opusclip-api --image gcr.io/your-project/video-opusclip-api:latest

# Azure Container Instances
az container create --resource-group your-rg --name video-opusclip-api --image your-registry.azurecr.io/video-opusclip-api:latest
```

---

## 📈 **Performance Optimization**

### **1. Horizontal Scaling**

```yaml
# kubernetes-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: video-opusclip-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: video-opusclip-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### **2. Database Optimization**

```sql
-- Create indexes for better performance
CREATE INDEX idx_video_requests_youtube_url ON video_requests(youtube_url);
CREATE INDEX idx_video_requests_created_at ON video_requests(created_at);
CREATE INDEX idx_video_requests_status ON video_requests(status);

-- Optimize queries
EXPLAIN ANALYZE SELECT * FROM video_requests WHERE youtube_url = '...';
```

### **3. Cache Optimization**

```python
# cache_optimization.py
from redis import Redis
import json

class OptimizedCacheManager:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.pipeline = self.redis.pipeline()
    
    async def batch_set(self, items: Dict[str, Any], ttl: int = 3600):
        """Batch set multiple items for better performance."""
        for key, value in items.items():
            self.pipeline.setex(key, ttl, json.dumps(value))
        await self.pipeline.execute()
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple items for better performance."""
        values = await self.pipeline.mget(keys)
        return {key: json.loads(value) for key, value in zip(keys, values) if value}
```

---

## 🎯 **Production Checklist**

### **Pre-Deployment**

- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance testing done
- [ ] Environment variables configured
- [ ] SSL certificates ready
- [ ] Database migrations applied
- [ ] Monitoring configured
- [ ] Backup strategy in place

### **Post-Deployment**

- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Logs being collected
- [ ] Alerts configured
- [ ] Performance monitoring active
- [ ] Security monitoring active
- [ ] Backup verification
- [ ] Documentation updated

---

## 🆘 **Troubleshooting**

### **Common Issues**

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   kubectl top pods
   
   # Optimize memory settings
   export WORKERS=2
   export MAX_WORKERS=4
   ```

2. **Database Connection Issues**
   ```bash
   # Check database connectivity
   psql $DATABASE_URL -c "SELECT 1;"
   
   # Check connection pool
   SELECT * FROM pg_stat_activity;
   ```

3. **Redis Connection Issues**
   ```bash
   # Test Redis connection
   redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
   
   # Check Redis memory
   redis-cli info memory
   ```

4. **Performance Issues**
   ```bash
   # Check application metrics
   curl http://localhost:8000/metrics
   
   # Check system resources
   htop
   iostat
   ```

---

## 🎉 **Deployment Complete!**

Your improved Video-OpusClip API is now deployed to production with:

- ✅ **High Availability**: Load balancing and auto-scaling
- ✅ **Performance**: Caching and optimization
- ✅ **Security**: SSL, rate limiting, and security headers
- ✅ **Monitoring**: Comprehensive metrics and alerting
- ✅ **Scalability**: Horizontal scaling capabilities
- ✅ **Reliability**: Health checks and error handling

**🚀 Your API is ready to handle production traffic!**






























