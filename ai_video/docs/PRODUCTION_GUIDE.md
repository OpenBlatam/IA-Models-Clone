# AI Video System - Production Guide

## 🚀 Production Deployment Guide

This guide provides comprehensive instructions for deploying the AI Video System in production environments with best practices, security considerations, and operational procedures.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Security](#security)
5. [Deployment](#deployment)
6. [Monitoring](#monitoring)
7. [Scaling](#scaling)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## 🔧 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: Minimum 10GB free space
- **CPU**: 2+ cores recommended
- **Network**: Stable internet connection

### Operating System

- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **Windows**: Windows 10/11, Windows Server 2019+
- **macOS**: macOS 10.15+

### Dependencies

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y build-essential libssl-dev libffi-dev
sudo apt install -y git curl wget

# Install system dependencies (CentOS/RHEL)
sudo yum install -y python3 python3-pip
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel libffi-devel
```

## 📦 Installation

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv ai_video_env
source ai_video_env/bin/activate  # Linux/macOS
# or
ai_video_env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements_production.txt

# Install optional dependencies as needed
# pip install -r requirements_unified.txt
```

### 3. System Setup

```bash
# Run installation script
python install.py --all

# Verify installation
python test_system.py --all
```

## ⚙️ Configuration

### 1. Environment Configuration

Create `.env` file:

```bash
# Copy template
cp config/.env.template .env

# Edit configuration
nano .env
```

Example `.env` configuration:

```bash
# System configuration
AI_VIDEO_ENVIRONMENT=production
AI_VIDEO_DEBUG=false
AI_VIDEO_LOG_LEVEL=INFO

# Storage configuration
AI_VIDEO_STORAGE_PATH=/var/ai_video/storage
AI_VIDEO_TEMP_DIR=/var/ai_video/temp
AI_VIDEO_OUTPUT_DIR=/var/ai_video/output

# Security configuration
AI_VIDEO_SECRET_KEY=your-secret-key-here
AI_VIDEO_ENABLE_AUTH=true
AI_VIDEO_ENABLE_RATE_LIMITING=true

# Performance configuration
AI_VIDEO_MAX_CONCURRENT_WORKFLOWS=10
AI_VIDEO_WORKFLOW_TIMEOUT=600

# Monitoring configuration
AI_VIDEO_ENABLE_METRICS=true
AI_VIDEO_METRICS_PORT=9090
```

### 2. Configuration Files

Create production configuration:

```bash
# Create production config
python main.py --create-config production_config.json

# Edit configuration
nano production_config.json
```

Example production configuration:

```json
{
  "environment": "production",
  "debug": false,
  "plugins": {
    "auto_discover": true,
    "auto_load": true,
    "validation_level": "strict",
    "plugin_dirs": ["/var/ai_video/plugins"],
    "enable_events": true,
    "enable_metrics": true
  },
  "workflow": {
    "max_concurrent_workflows": 10,
    "workflow_timeout": 600,
    "enable_retry": true,
    "max_retries": 3,
    "extraction_timeout": 120,
    "max_content_length": 100000,
    "enable_language_detection": true,
    "default_duration": 30.0,
    "default_resolution": "1920x1080",
    "default_quality": "high",
    "enable_avatar_selection": true,
    "enable_caching": true,
    "cache_ttl": 3600,
    "enable_metrics": true,
    "enable_monitoring": true
  },
  "ai": {
    "default_model": "gpt-4",
    "fallback_model": "gpt-3.5-turbo",
    "max_tokens": 4000,
    "temperature": 0.7,
    "api_timeout": 60,
    "api_retries": 3,
    "enable_streaming": false,
    "enable_content_optimization": true,
    "enable_short_video_optimization": true,
    "enable_langchain_analysis": true,
    "suggestion_count": 3,
    "enable_music_suggestions": true,
    "enable_visual_suggestions": true,
    "enable_transition_suggestions": true
  },
  "storage": {
    "local_storage_path": "/var/ai_video/storage",
    "temp_directory": "/var/ai_video/temp",
    "output_directory": "/var/ai_video/output",
    "max_file_size": 104857600,
    "allowed_formats": ["mp4", "avi", "mov", "mkv"],
    "enable_compression": true,
    "auto_cleanup": true,
    "cleanup_interval": 86400,
    "max_age_days": 30
  },
  "security": {
    "enable_auth": true,
    "auth_token_expiry": 3600,
    "enable_url_validation": true,
    "allowed_domains": ["example.com", "trusted-site.com"],
    "blocked_domains": ["malicious-site.com"],
    "enable_content_filtering": true,
    "filter_inappropriate_content": true,
    "enable_nsfw_detection": false,
    "enable_rate_limiting": true,
    "max_requests_per_minute": 60,
    "max_requests_per_hour": 1000
  },
  "monitoring": {
    "log_level": "INFO",
    "log_file": "/var/log/ai_video/ai_video.log",
    "enable_structured_logging": true,
    "enable_metrics": true,
    "metrics_port": 9090,
    "enable_prometheus": true,
    "enable_health_checks": true,
    "health_check_interval": 300,
    "enable_alerts": true,
    "alert_webhook_url": "https://your-alerting-service.com/webhook"
  }
}
```

## 🔒 Security

### 1. Access Control

```bash
# Create dedicated user
sudo useradd -r -s /bin/false ai_video
sudo usermod -aG ai_video www-data

# Set proper permissions
sudo chown -R ai_video:ai_video /var/ai_video
sudo chmod -R 750 /var/ai_video
sudo chmod 640 /var/ai_video/config/*.json
```

### 2. Network Security

```bash
# Configure firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 9090/tcp  # Metrics
sudo ufw enable

# Configure SSL/TLS
sudo apt install -y certbot
sudo certbot --nginx -d your-domain.com
```

### 3. API Security

```python
# Example API security configuration
from ai_video import AIVideoSystem

system = AIVideoSystem("production_config.json")

# Enable authentication
system.config.security.enable_auth = True
system.config.security.auth_token_expiry = 3600

# Enable rate limiting
system.config.security.enable_rate_limiting = True
system.config.security.max_requests_per_minute = 60

# Enable content filtering
system.config.security.enable_content_filtering = True
system.config.security.filter_inappropriate_content = True
```

## 🚀 Deployment

### 1. Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -r -s /bin/false ai_video

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /var/ai_video/storage /var/ai_video/temp /var/ai_video/output /var/log/ai_video

# Set permissions
RUN chown -R ai_video:ai_video /app /var/ai_video /var/log/ai_video

# Switch user
USER ai_video

# Expose ports
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:9090/health')"

# Start application
CMD ["python", "main.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ai_video:
    build: .
    ports:
      - "9090:9090"
    volumes:
      - ./config:/app/config
      - ./storage:/var/ai_video/storage
      - ./logs:/var/log/ai_video
    environment:
      - AI_VIDEO_ENVIRONMENT=production
      - AI_VIDEO_CONFIG_FILE=/app/config/production_config.json
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:9090/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai_video
    restart: unless-stopped
```

### 2. Systemd Service

Create `/etc/systemd/system/ai-video.service`:

```ini
[Unit]
Description=AI Video System
After=network.target

[Service]
Type=simple
User=ai_video
Group=ai_video
WorkingDirectory=/var/ai_video
Environment=AI_VIDEO_ENVIRONMENT=production
Environment=AI_VIDEO_CONFIG_FILE=/var/ai_video/config/production_config.json
ExecStart=/var/ai_video/venv/bin/python main.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-video
sudo systemctl start ai-video
sudo systemctl status ai-video
```

### 3. Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-video
  labels:
    app: ai-video
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-video
  template:
    metadata:
      labels:
        app: ai-video
    spec:
      containers:
      - name: ai-video
        image: ai-video:latest
        ports:
        - containerPort: 9090
        env:
        - name: AI_VIDEO_ENVIRONMENT
          value: "production"
        - name: AI_VIDEO_CONFIG_FILE
          value: "/app/config/production_config.json"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: storage
          mountPath: /var/ai_video/storage
        - name: logs
          mountPath: /var/log/ai_video
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: ai-video-config
      - name: storage
        persistentVolumeClaim:
          claimName: ai-video-storage
      - name: logs
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: ai-video-service
spec:
  selector:
    app: ai-video
  ports:
  - port: 80
    targetPort: 9090
  type: LoadBalancer
```

## 📊 Monitoring

### 1. Health Checks

```bash
# Check system health
curl http://localhost:9090/health

# Check system status
curl http://localhost:9090/status

# Get metrics
curl http://localhost:9090/metrics
```

### 2. Logging

```bash
# View logs
tail -f /var/log/ai_video/ai_video.log

# Search for errors
grep ERROR /var/log/ai_video/ai_video.log

# Monitor in real-time
journalctl -u ai-video -f
```

### 3. Prometheus Integration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-video'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 4. Grafana Dashboard

Create dashboard configuration:

```json
{
  "dashboard": {
    "title": "AI Video System",
    "panels": [
      {
        "title": "Workflow Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ai_video_workflow_success_rate[5m])"
          }
        ]
      },
      {
        "title": "Workflow Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ai_video_workflow_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Plugin Usage",
        "type": "table",
        "targets": [
          {
            "expr": "ai_video_plugin_usage_total"
          }
        ]
      }
    ]
  }
}
```

## 📈 Scaling

### 1. Horizontal Scaling

```bash
# Scale Docker Compose
docker-compose up --scale ai_video=5

# Scale Kubernetes
kubectl scale deployment ai-video --replicas=10

# Scale systemd (multiple instances)
sudo systemctl start ai-video@1
sudo systemctl start ai-video@2
sudo systemctl start ai-video@3
```

### 2. Load Balancing

Configure Nginx load balancer:

```nginx
upstream ai_video_backend {
    least_conn;
    server 127.0.0.1:9091;
    server 127.0.0.1:9092;
    server 127.0.0.1:9093;
    server 127.0.0.1:9094;
    server 127.0.0.1:9095;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://ai_video_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Database Scaling

```python
# Configure Redis for caching
import redis

redis_client = redis.Redis(
    host='redis-cluster.example.com',
    port=6379,
    db=0,
    decode_responses=True
)

# Configure database connection pooling
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://user:pass@db-cluster.example.com:5432/ai_video',
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)
```

## 💾 Backup & Recovery

### 1. Data Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/ai_video"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /var/ai_video/config/

# Backup storage
tar -czf $BACKUP_DIR/storage_$DATE.tar.gz /var/ai_video/storage/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /var/log/ai_video/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### 2. Database Backup

```bash
# PostgreSQL backup
pg_dump ai_video > /backup/database_$DATE.sql

# Redis backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb /backup/redis_$DATE.rdb
```

### 3. Recovery Procedures

```bash
#!/bin/bash
# recovery.sh

BACKUP_DATE=$1
BACKUP_DIR="/backup/ai_video"

# Stop services
sudo systemctl stop ai-video

# Restore configuration
tar -xzf $BACKUP_DIR/config_$BACKUP_DATE.tar.gz -C /

# Restore storage
tar -xzf $BACKUP_DIR/storage_$BACKUP_DATE.tar.gz -C /

# Restore logs
tar -xzf $BACKUP_DIR/logs_$BACKUP_DATE.tar.gz -C /

# Start services
sudo systemctl start ai-video
```

## 🔧 Troubleshooting

### 1. Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Restart service
sudo systemctl restart ai-video

# Increase memory limits
# Edit production_config.json
{
  "workflow": {
    "max_concurrent_workflows": 5  # Reduce from 10
  }
}
```

#### High CPU Usage
```bash
# Check CPU usage
top
htop

# Check for stuck processes
ps aux | grep python

# Kill stuck processes
pkill -f "ai_video"
```

#### Disk Space Issues
```bash
# Check disk usage
df -h
du -sh /var/ai_video/*

# Clean up old files
python -c "from ai_video.core.utils import cleanup_old_files; cleanup_old_files('/var/ai_video/storage', 7)"

# Increase storage
# Add more storage or configure external storage
```

### 2. Log Analysis

```bash
# Find errors
grep -i error /var/log/ai_video/ai_video.log

# Find slow operations
grep "took.*s" /var/log/ai_video/ai_video.log | sort -k2 -n

# Monitor real-time
tail -f /var/log/ai_video/ai_video.log | grep -E "(ERROR|WARNING|CRITICAL)"
```

### 3. Performance Tuning

```python
# Optimize configuration
config = {
    "workflow": {
        "max_concurrent_workflows": 5,  # Adjust based on CPU cores
        "workflow_timeout": 300,        # Reduce for faster failures
        "enable_caching": True,         # Enable caching
        "cache_ttl": 1800              # 30 minutes cache
    },
    "storage": {
        "enable_compression": True,     # Enable compression
        "auto_cleanup": True,          # Enable auto cleanup
        "max_age_days": 7              # Keep files for 7 days
    }
}
```

## ✅ Best Practices

### 1. Security Best Practices

- ✅ Use dedicated user for running the service
- ✅ Set proper file permissions
- ✅ Enable authentication and authorization
- ✅ Use HTTPS in production
- ✅ Implement rate limiting
- ✅ Regular security updates
- ✅ Monitor for suspicious activity

### 2. Performance Best Practices

- ✅ Monitor resource usage
- ✅ Use connection pooling
- ✅ Implement caching
- ✅ Optimize database queries
- ✅ Use async/await properly
- ✅ Implement circuit breakers
- ✅ Use load balancing

### 3. Monitoring Best Practices

- ✅ Set up comprehensive logging
- ✅ Monitor key metrics
- ✅ Set up alerting
- ✅ Use health checks
- ✅ Monitor external dependencies
- ✅ Track business metrics
- ✅ Regular performance reviews

### 4. Deployment Best Practices

- ✅ Use version control
- ✅ Implement CI/CD pipelines
- ✅ Use containerization
- ✅ Implement blue-green deployments
- ✅ Use configuration management
- ✅ Implement rollback procedures
- ✅ Test in staging environment

### 5. Operational Best Practices

- ✅ Document procedures
- ✅ Train operations team
- ✅ Regular backups
- ✅ Disaster recovery planning
- ✅ Capacity planning
- ✅ Regular maintenance windows
- ✅ Incident response procedures

## 📞 Support

### Getting Help

1. **Documentation**: Check the README and system documentation
2. **Logs**: Review application and system logs
3. **Health Checks**: Run system health checks
4. **Community**: Check GitHub issues and discussions
5. **Professional Support**: Contact the development team

### Emergency Procedures

1. **Service Down**: Restart the service
2. **High Resource Usage**: Scale down or restart
3. **Data Loss**: Restore from backup
4. **Security Breach**: Isolate and investigate
5. **Performance Issues**: Check monitoring and optimize

---

**Remember**: Always test changes in a staging environment before applying to production! 