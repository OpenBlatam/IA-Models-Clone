# Enhanced Blaze AI Deployment Guide

This guide provides comprehensive instructions for deploying and using the enhanced Blaze AI system with all its enterprise-grade features.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Testing Enhanced Features](#testing-enhanced-features)
6. [Production Deployment](#production-deployment)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Security Configuration](#security-configuration)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tuning](#performance-tuning)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: Minimum 10GB free space
- **CPU**: 2+ cores recommended
- **OS**: Linux, macOS, or Windows

### Dependencies

- **Redis**: For distributed rate limiting and caching
- **PostgreSQL/MySQL**: For persistent data storage (optional)
- **Docker**: For containerized deployment (optional)
- **Kubernetes**: For orchestrated deployment (optional)

### Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation

### 1. Clone and Install Dependencies

```bash
# Navigate to the Blaze AI directory
cd agents/backend/onyx/server/features/blaze_ai

# Install enhanced dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, uvicorn, redis, prometheus_client; print('Dependencies installed successfully!')"
```

### 2. Environment Setup

Create a `.env` file with your configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your values
nano .env
```

Example `.env` file:

```env
# Core Configuration
APP_ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
JWT_SECRET_KEY=your-super-secret-key-change-in-production
API_KEY_REQUIRED=true

# External Services
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
STABILITY_API_KEY=your-stability-api-key

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost:5432/blaze_ai

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
```

### 3. Redis Setup

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify Redis is running
redis-cli ping
# Should return: PONG

# Test Redis connection
redis-cli set test "Hello Blaze AI"
redis-cli get test
```

## Configuration

### 1. Basic Configuration

The system uses `config-enhanced.yaml` for comprehensive configuration. Key sections:

```yaml
# Security Configuration
security:
  enable_authentication: true
  enable_authorization: true
  enable_threat_detection: true
  
  jwt:
    secret_key: "your-secret-key"
    expiration: 3600

# Rate Limiting
rate_limiting:
  algorithm: "adaptive"
  requests_per_minute: 100
  requests_per_hour: 1000

# Performance Monitoring
monitoring:
  enable_monitoring: true
  enable_profiling: true
  enable_alerting: true
```

### 2. Security Configuration

```yaml
security:
  threat_detection:
    enable_sql_injection_detection: true
    enable_xss_detection: true
    enable_path_traversal_detection: true
    max_failed_attempts: 5
    lockout_duration: 300
    enable_ip_blacklisting: true
```

### 3. Monitoring Configuration

```yaml
monitoring:
  alert_thresholds:
    system:
      cpu:
        percent: 80.0
      memory:
        percent: 85.0
      disk:
        percent: 90.0
  
  prometheus:
    enable: true
    port: 9090
    path: "/metrics"
```

## Running the Application

### 1. Development Mode

```bash
# Start with hot reload
python main.py --dev

# Or start manually
uvicorn main:create_app --host 0.0.0.0 --port 8000 --reload
```

### 2. Production Mode

```bash
# Start production server
python main.py

# Or with uvicorn
uvicorn main:create_app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Docker Deployment

```bash
# Build Docker image
docker build -t blaze-ai:latest .

# Run container
docker run -d \
  --name blaze-ai \
  -p 8000:8000 \
  -p 9090:9090 \
  -e REDIS_HOST=host.docker.internal \
  blaze-ai:latest

# Check logs
docker logs blaze-ai
```

### 4. Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n blaze-ai
kubectl get services -n blaze-ai
```

## Testing Enhanced Features

### 1. Run the Test Suite

```bash
# Install test dependencies
pip install requests pytest

# Run comprehensive tests
python test_enhanced_features.py

# View test report
cat enhanced_features_test_report.md
```

### 2. Manual Testing

#### Health Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Metrics
curl http://localhost:8000/metrics

# Prometheus metrics
curl http://localhost:8000/metrics/prometheus

# Security status
curl http://localhost:8000/security/status

# Error summary
curl http://localhost:8000/errors/summary
```

#### API Documentation

```bash
# Swagger UI
open http://localhost:8000/docs

# ReDoc
open http://localhost:8000/redoc

# OpenAPI JSON
curl http://localhost:8000/openapi.json
```

### 3. Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test rate limiting
ab -n 1000 -c 10 http://localhost:8000/health

# Test with authentication
ab -n 100 -c 5 -H "X-API-Key: your-api-key" http://localhost:8000/health
```

## Production Deployment

### 1. Environment Variables

```bash
# Production environment
export APP_ENVIRONMENT=production
export DEBUG=false
export JWT_SECRET_KEY=your-production-secret-key
export REDIS_HOST=your-redis-host
export DATABASE_URL=your-production-database-url
```

### 2. Systemd Service

Create `/etc/systemd/system/blaze-ai.service`:

```ini
[Unit]
Description=Blaze AI Enhanced Service
After=network.target redis.service

[Service]
Type=simple
User=blaze-ai
WorkingDirectory=/opt/blaze-ai
Environment=PATH=/opt/blaze-ai/venv/bin
ExecStart=/opt/blaze-ai/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable blaze-ai
sudo systemctl start blaze-ai
sudo systemctl status blaze-ai
```

### 3. Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /metrics {
        proxy_pass http://127.0.0.1:8000;
        auth_basic "Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

### 4. SSL/TLS with Let's Encrypt

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring and Observability

### 1. Prometheus Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'blaze-ai'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
    scrape_interval: 5s
```

### 2. Grafana Dashboard

Import the provided Grafana dashboard JSON or create custom dashboards for:

- System metrics (CPU, memory, disk)
- Application metrics (request rate, response time, error rate)
- Business metrics (AI requests, model usage)
- Security metrics (threats blocked, failed attempts)

### 3. Alerting

Configure alerts for:

```yaml
# High CPU usage
- alert: HighCPUUsage
  expr: system_cpu_percent > 80
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High CPU usage detected"

# High error rate
- alert: HighErrorRate
  expr: rate(ai_errors_total[5m]) > 0.1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
```

## Security Configuration

### 1. Authentication

```python
# JWT Configuration
JWT_SECRET_KEY = "your-secure-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600

# API Key Configuration
API_KEY_HEADER = "X-API-Key"
API_KEY_REQUIRED = True
```

### 2. Authorization

```python
# Role-based access control
ROLES = {
    "admin": ["read", "write", "delete", "admin"],
    "user": ["read", "write"],
    "viewer": ["read"]
}

# Permission mapping
PERMISSIONS = {
    "/admin": ["admin"],
    "/api/v2": ["user", "admin"],
    "/metrics": ["admin"]
}
```

### 3. Threat Detection

```yaml
security:
  threat_detection:
    enable_sql_injection_detection: true
    enable_xss_detection: true
    enable_path_traversal_detection: true
    enable_command_injection_detection: true
    
    suspicious_patterns:
      - "eval\\s*\\("
      - "exec\\s*\\("
      - "system\\s*\\("
      - "subprocess\\s*\\("
    
    max_failed_attempts: 5
    lockout_duration: 300
    enable_ip_blacklisting: true
```

## Troubleshooting

### 1. Common Issues

#### Service Won't Start

```bash
# Check logs
sudo journalctl -u blaze-ai -f

# Check dependencies
python -c "import fastapi, redis, prometheus_client"

# Check configuration
python -c "import yaml; yaml.safe_load(open('config-enhanced.yaml'))"
```

#### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping

# Check Redis status
sudo systemctl status redis-server

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log
```

#### Performance Issues

```bash
# Check system resources
htop
iostat -x 1
netstat -i

# Check application metrics
curl http://localhost:8000/metrics

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=system_cpu_percent
```

### 2. Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Start with debug output
python main.py --dev

# Check detailed logs
tail -f logs/blaze_ai.log
```

### 3. Health Checks

```bash
# Check all health endpoints
curl http://localhost:8000/health/detailed | jq

# Check specific systems
curl http://localhost:8000/health/detailed | jq '.systems'

# Check error summary
curl http://localhost:8000/errors/summary | jq
```

## Performance Tuning

### 1. System Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" >> /etc/sysctl.conf
sysctl -p
```

### 2. Application Tuning

```yaml
# Performance configuration
performance:
  workers: 4
  max_connections: 1000
  keepalive_timeout: 65
  max_requests: 1000
  max_requests_jitter: 100
  
  # Connection pooling
  pool_size: 20
  max_overflow: 30
  
  # Caching
  cache_ttl: 3600
  cache_max_size: 10000
```

### 3. Database Optimization

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Reload configuration
SELECT pg_reload_conf();
```

### 4. Monitoring and Alerts

```yaml
# Performance alerts
alerts:
  - name: "HighResponseTime"
    condition: "response_time_p95 > 1000ms"
    duration: "5m"
    severity: "warning"
    
  - name: "HighErrorRate"
    condition: "error_rate > 5%"
    duration: "2m"
    severity: "critical"
    
  - name: "HighMemoryUsage"
    condition: "memory_usage > 85%"
    duration: "5m"
    severity: "warning"
```

## Support and Maintenance

### 1. Regular Maintenance

```bash
# Daily health checks
curl -f http://localhost:8000/health || echo "Service down!"

# Weekly metrics review
curl http://localhost:8000/metrics | jq '.system'

# Monthly security audit
curl http://localhost:8000/security/status | jq
```

### 2. Backup and Recovery

```bash
# Database backup
pg_dump blaze_ai > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz *.yaml *.env

# Log rotation
logrotate /etc/logrotate.d/blaze-ai
```

### 3. Updates and Upgrades

```bash
# Check for updates
pip list --outdated

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart blaze-ai

# Verify update
curl http://localhost:8000/health
```

## Conclusion

This enhanced Blaze AI system provides enterprise-grade features including:

- **Advanced Security**: Authentication, authorization, threat detection
- **Performance Monitoring**: Real-time metrics, profiling, alerting
- **Rate Limiting**: Multi-algorithm, distributed, adaptive
- **Error Handling**: Circuit breakers, retry logic, recovery strategies
- **Observability**: Health checks, metrics export, comprehensive logging

Follow this guide to deploy and maintain a production-ready Blaze AI system that can handle enterprise workloads with confidence.

For additional support or questions, refer to the project documentation or create an issue in the project repository.
