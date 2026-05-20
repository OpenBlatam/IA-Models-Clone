# BUL Enhanced API - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the BUL Enhanced API in various environments, from development to production.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Memory**: Minimum 2GB RAM (4GB+ recommended for production)
- **Storage**: 10GB+ free space
- **Network**: Internet connection for external API calls

### External Dependencies

- **PostgreSQL**: 13+ (for production database)
- **Redis**: 6+ (for caching)
- **OpenRouter API Key**: For AI model access
- **OpenAI API Key**: Optional, for fallback

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd bul-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file:

```bash
# API Configuration
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/bul_db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your_secret_key_here_minimum_32_characters
JWT_ALGORITHM=HS256

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
DEBUG=false
ENVIRONMENT=production

# CORS Configuration
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### 3. Database Setup

```bash
# Install PostgreSQL and Redis
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib redis-server

# macOS:
brew install postgresql redis

# Create database
sudo -u postgres createdb bul_db
sudo -u postgres createuser bul_user
sudo -u postgres psql -c "ALTER USER bul_user PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bul_db TO bul_user;"
```

### 4. Run the Application

```bash
# Development
python main.py

# Production with Gunicorn
gunicorn api.enhanced_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "api.enhanced_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://bul_user:bul_password@db:5432/bul_db
      - REDIS_URL=redis://redis:6379/0
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: bul_db
      POSTGRES_USER: bul_user
      POSTGRES_PASSWORD: bul_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bul_user -d bul_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

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

volumes:
  postgres_data:
  redis_data:
```

### 3. Nginx Configuration

```nginx
events {
    worker_connections 1024;
}

http {
    upstream bul_api {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Rate limiting
        limit_req zone=api burst=20 nodelay;

        location / {
            proxy_pass http://bul_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://bul_api/health;
            access_log off;
        }
    }
}
```

## Kubernetes Deployment

### 1. Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: bul-api
```

### 2. ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bul-api-config
  namespace: bul-api
data:
  DATABASE_URL: "postgresql://bul_user:bul_password@postgres-service:5432/bul_db"
  REDIS_URL: "redis://redis-service:6379/0"
  HOST: "0.0.0.0"
  PORT: "8000"
  WORKERS: "4"
  DEBUG: "false"
  ENVIRONMENT: "production"
```

### 3. Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: bul-api-secrets
  namespace: bul-api
type: Opaque
data:
  OPENROUTER_API_KEY: <base64-encoded-key>
  OPENAI_API_KEY: <base64-encoded-key>
  SECRET_KEY: <base64-encoded-secret>
```

### 4. Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bul-api
  namespace: bul-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bul-api
  template:
    metadata:
      labels:
        app: bul-api
    spec:
      containers:
      - name: bul-api
        image: bul-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: bul-api-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: bul-api-config
              key: REDIS_URL
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: bul-api-secrets
              key: OPENROUTER_API_KEY
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: bul-api-secrets
              key: SECRET_KEY
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 5. Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: bul-api-service
  namespace: bul-api
spec:
  selector:
    app: bul-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### 6. Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bul-api-ingress
  namespace: bul-api
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: bul-api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bul-api-service
            port:
              number: 80
```

## Production Deployment

### 1. Environment Setup

```bash
# Create production environment
python -m venv venv-prod
source venv-prod/bin/activate
pip install -r requirements-prod.txt

# Set production environment variables
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO
```

### 2. Database Migration

```bash
# Run database migrations
python -m alembic upgrade head

# Create initial data
python scripts/init_data.py
```

### 3. SSL Certificate

```bash
# Using Let's Encrypt
sudo certbot --nginx -d yourdomain.com

# Or using custom certificates
sudo cp your-cert.pem /etc/ssl/certs/
sudo cp your-key.pem /etc/ssl/private/
```

### 4. Process Management

```bash
# Using systemd
sudo cp bul-api.service /etc/systemd/system/
sudo systemctl enable bul-api
sudo systemctl start bul-api
sudo systemctl status bul-api
```

### 5. Monitoring Setup

```bash
# Install monitoring tools
pip install prometheus-client
pip install sentry-sdk

# Configure monitoring
export SENTRY_DSN=your_sentry_dsn
export PROMETHEUS_PORT=9090
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_document_records_created_at ON document_records(created_at);
CREATE INDEX idx_document_records_user_id ON document_records(user_id);
CREATE INDEX idx_api_logs_created_at ON api_logs(created_at);
CREATE INDEX idx_api_logs_user_id ON api_logs(user_id);

-- Optimize PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

### 2. Redis Optimization

```bash
# Redis configuration
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 3. Application Optimization

```python
# Use connection pooling
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30

# Enable compression
ENABLE_COMPRESSION = True

# Cache settings
CACHE_TTL = 3600
CACHE_MAX_SIZE = 1000

# Rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60
```

## Security Hardening

### 1. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw enable
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 5432/tcp  # Block direct database access
sudo ufw deny 6379/tcp  # Block direct Redis access
```

### 2. Database Security

```sql
-- Create restricted database user
CREATE USER bul_app_user WITH PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE bul_db TO bul_app_user;
GRANT USAGE ON SCHEMA public TO bul_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO bul_app_user;
```

### 3. Application Security

```python
# Security settings
SECRET_KEY = "your-very-long-secret-key-here"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60

# CORS settings
CORS_ORIGINS = ["https://yourdomain.com"]
CORS_ALLOW_CREDENTIALS = True
```

## Monitoring and Logging

### 1. Logging Configuration

```python
# Logging setup
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/bul-api/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
```

### 2. Health Monitoring

```bash
# Health check script
#!/bin/bash
curl -f http://localhost:8000/health || exit 1
```

### 3. Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('bul_api_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('bul_api_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('bul_api_active_connections', 'Active connections')
```

## Backup and Recovery

### 1. Database Backup

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U bul_user bul_db > /backups/bul_db_$DATE.sql
gzip /backups/bul_db_$DATE.sql

# Keep only last 7 days
find /backups -name "bul_db_*.sql.gz" -mtime +7 -delete
```

### 2. Application Backup

```bash
# Backup application data
tar -czf /backups/bul_api_$DATE.tar.gz /app/data/
```

### 3. Recovery Procedure

```bash
# Database recovery
gunzip /backups/bul_db_$DATE.sql.gz
psql -h localhost -U bul_user bul_db < /backups/bul_db_$DATE.sql

# Application recovery
tar -xzf /backups/bul_api_$DATE.tar.gz -C /
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check database status
   sudo systemctl status postgresql
   
   # Check connection
   psql -h localhost -U bul_user -d bul_db
   ```

2. **Redis Connection Errors**
   ```bash
   # Check Redis status
   sudo systemctl status redis
   
   # Test connection
   redis-cli ping
   ```

3. **API Not Responding**
   ```bash
   # Check application logs
   tail -f /var/log/bul-api/app.log
   
   # Check health endpoint
   curl http://localhost:8000/health
   ```

4. **High Memory Usage**
   ```bash
   # Check memory usage
   free -h
   ps aux --sort=-%mem
   
   # Check for memory leaks
   python -m memory_profiler app.py
   ```

### Performance Issues

1. **Slow Response Times**
   - Check database query performance
   - Enable Redis caching
   - Optimize API endpoints
   - Check external API response times

2. **High CPU Usage**
   - Check for infinite loops
   - Optimize algorithms
   - Scale horizontally
   - Check for blocking operations

3. **Memory Leaks**
   - Monitor memory usage over time
   - Check for circular references
   - Use memory profiling tools
   - Restart application periodically

## Maintenance

### 1. Regular Maintenance Tasks

```bash
# Daily tasks
- Check application health
- Monitor resource usage
- Review error logs
- Backup database

# Weekly tasks
- Update dependencies
- Clean old logs
- Analyze performance metrics
- Security updates

# Monthly tasks
- Full system backup
- Security audit
- Performance review
- Capacity planning
```

### 2. Update Procedure

```bash
# Backup current version
cp -r /app /app.backup

# Update application
git pull origin main
pip install -r requirements.txt

# Run migrations
python -m alembic upgrade head

# Restart application
sudo systemctl restart bul-api

# Verify deployment
curl http://localhost:8000/health
```

### 3. Rollback Procedure

```bash
# Stop application
sudo systemctl stop bul-api

# Restore backup
rm -rf /app
mv /app.backup /app

# Restart application
sudo systemctl start bul-api

# Verify rollback
curl http://localhost:8000/health
```

## Support and Documentation

- **API Documentation**: `/docs` (Swagger UI)
- **ReDoc Documentation**: `/redoc`
- **Health Check**: `/health`
- **OpenAPI Schema**: `/openapi.json`

For additional support, please refer to the main documentation or contact the development team.