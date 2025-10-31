# AI Integration System - Deployment Guide

## 🚀 Quick Start Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Clone and navigate to the system
cd ai_integration_system

# Copy and configure environment
cp config_template.env .env
# Edit .env with your actual configuration

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f ai-integration-api
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config_template.env .env
# Edit .env with your configuration

# Initialize database
python -c "from database import create_tables; create_tables()"

# Start the application
python start.py
```

## 📋 Pre-Deployment Checklist

### 1. Environment Configuration

- [ ] Copy `config_template.env` to `.env`
- [ ] Configure database connection string
- [ ] Configure Redis connection string
- [ ] Set secure secret key
- [ ] Configure platform credentials (Salesforce, Mailchimp, etc.)
- [ ] Set appropriate log level
- [ ] Configure monitoring settings

### 2. Platform Setup

#### Salesforce Setup
```env
SALESFORCE__ENABLED=true
SALESFORCE__BASE_URL=https://your-instance.salesforce.com
SALESFORCE__CLIENT_ID=your_connected_app_client_id
SALESFORCE__CLIENT_SECRET=your_connected_app_secret
SALESFORCE__USERNAME=your_username
SALESFORCE__PASSWORD=your_password
SALESFORCE__SECURITY_TOKEN=your_security_token
```

#### Mailchimp Setup
```env
MAILCHIMP__ENABLED=true
MAILCHIMP__API_KEY=your_api_key
MAILCHIMP__SERVER_PREFIX=us1
MAILCHIMP__LIST_ID=your_list_id
```

#### WordPress Setup
```env
WORDPRESS__ENABLED=true
WORDPRESS__BASE_URL=https://your-site.com
WORDPRESS__USERNAME=your_username
WORDPRESS__APPLICATION_PASSWORD=your_app_password
```

#### HubSpot Setup
```env
HUBSPOT__ENABLED=true
HUBSPOT__API_KEY=your_api_key
HUBSPOT__PORTAL_ID=your_portal_id
```

### 3. Database Setup

#### PostgreSQL (Recommended)
```bash
# Create database
createdb ai_integration

# Set environment variable
export DATABASE_URL=postgresql://user:password@localhost:5432/ai_integration
```

#### SQLite (Development)
```bash
# Set environment variable
export DATABASE_URL=sqlite:///ai_integration.db
```

### 4. Redis Setup

```bash
# Install Redis
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server

# Set environment variable
export REDIS_URL=redis://localhost:6379/0
```

## 🐳 Docker Deployment

### Production Docker Compose

```yaml
version: '3.8'

services:
  ai-integration-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/ai_integration
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  ai-integration-worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/ai_integration
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ai_integration
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  postgres_data:
```

### Environment Variables for Docker

Create a `.env` file:
```env
POSTGRES_PASSWORD=your_secure_password
SECRET_KEY=your_secret_key
SALESFORCE__CLIENT_ID=your_client_id
SALESFORCE__CLIENT_SECRET=your_client_secret
# ... other platform configurations
```

## ☸️ Kubernetes Deployment

### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-integration
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-integration-config
  namespace: ai-integration
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-integration-secrets
  namespace: ai-integration
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  SECRET_KEY: <base64-encoded-secret-key>
  SALESFORCE__CLIENT_SECRET: <base64-encoded-secret>
  # ... other secrets
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-integration-api
  namespace: ai-integration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-integration-api
  template:
    metadata:
      labels:
        app: ai-integration-api
    spec:
      containers:
      - name: api
        image: ai-integration-system:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ai-integration-config
        - secretRef:
            name: ai-integration-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
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

### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-integration-service
  namespace: ai-integration
spec:
  selector:
    app: ai-integration-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔧 Production Configuration

### 1. Security Hardening

#### Environment Variables
```env
# Use strong, unique secret key
SECRET_KEY=your-very-long-random-secret-key-here

# Disable debug mode
DEBUG=false
ENVIRONMENT=production

# Use secure database credentials
DATABASE_URL=postgresql://secure_user:secure_password@db:5432/ai_integration

# Use secure Redis credentials
REDIS_URL=redis://:secure_password@redis:6379/0
```

#### SSL/TLS Configuration
```yaml
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://ai-integration-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Performance Optimization

#### Database Optimization
```sql
-- PostgreSQL optimizations
CREATE INDEX CONCURRENTLY idx_integration_requests_content_id ON integration_requests(content_id);
CREATE INDEX CONCURRENTLY idx_integration_requests_status ON integration_requests(status);
CREATE INDEX CONCURRENTLY idx_integration_requests_created_at ON integration_requests(created_at);

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

#### Redis Optimization
```conf
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 3. Monitoring and Logging

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-integration'
    static_configs:
      - targets: ['ai-integration-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

#### Log Aggregation
```yaml
# docker-compose.yml
services:
  ai-integration-api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 📊 Health Checks and Monitoring

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/ai-integration/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Monitoring Setup

1. **Prometheus**: Collect metrics
2. **Grafana**: Visualize metrics
3. **AlertManager**: Handle alerts
4. **ELK Stack**: Log aggregation

### Key Metrics to Monitor

- Integration success rate
- Response times
- Queue size
- Platform health
- System resources (CPU, memory, disk)
- Database connections
- Error rates

## 🔄 Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U postgres ai_integration > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U postgres ai_integration > $BACKUP_DIR/backup_$DATE.sql
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete
```

### Configuration Backup

```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz .env docker-compose.yml
```

### Recovery Procedures

1. **Database Recovery**:
   ```bash
   psql -h localhost -U postgres ai_integration < backup_file.sql
   ```

2. **Configuration Recovery**:
   ```bash
   tar -xzf config_backup_file.tar.gz
   ```

3. **Service Recovery**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database connectivity
python -c "from database import test_database_connection; print(test_database_connection())"

# Check database health
python -c "from database import check_database_health; print(check_database_health())"
```

#### 2. Platform Authentication Issues
```bash
# Test platform connections
curl -X POST "http://localhost:8000/ai-integration/platforms/salesforce/test"
curl -X POST "http://localhost:8000/ai-integration/platforms/mailchimp/test"
```

#### 3. Queue Processing Issues
```bash
# Check queue status
curl http://localhost:8000/ai-integration/queue/status

# Manually process queue
curl -X POST http://localhost:8000/ai-integration/queue/process
```

#### 4. Memory Issues
```bash
# Check system resources
docker stats

# Check application logs
docker-compose logs ai-integration-api
```

### Log Analysis

```bash
# View recent logs
docker-compose logs --tail=100 ai-integration-api

# Follow logs in real-time
docker-compose logs -f ai-integration-api

# Search for errors
docker-compose logs ai-integration-api | grep ERROR
```

## 📈 Scaling

### Horizontal Scaling

1. **API Scaling**:
   ```yaml
   # Increase replicas
   replicas: 5
   ```

2. **Worker Scaling**:
   ```yaml
   # Add more worker instances
   ai-integration-worker-2:
     # ... same configuration
   ```

3. **Database Scaling**:
   - Read replicas
   - Connection pooling
   - Query optimization

### Vertical Scaling

1. **Resource Limits**:
   ```yaml
   resources:
     requests:
       memory: "512Mi"
       cpu: "500m"
     limits:
       memory: "1Gi"
       cpu: "1000m"
   ```

2. **Database Resources**:
   - Increase memory allocation
   - Optimize query performance
   - Add indexes

## 🔐 Security Best Practices

1. **Environment Variables**:
   - Use secrets management
   - Rotate credentials regularly
   - Use least privilege principle

2. **Network Security**:
   - Use VPN for database access
   - Implement firewall rules
   - Use HTTPS everywhere

3. **Application Security**:
   - Regular security updates
   - Input validation
   - Rate limiting
   - Authentication and authorization

4. **Monitoring**:
   - Security event logging
   - Intrusion detection
   - Regular security audits

---

**This deployment guide provides comprehensive instructions for deploying the AI Integration System in various environments. Follow the steps carefully and adapt them to your specific infrastructure requirements.**



























