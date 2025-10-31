# 🚀 Email Sequence AI - Deployment Guide

This guide covers various deployment options for the Email Sequence AI system, from development to production.

## 📋 Prerequisites

### System Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB+ (8GB+ recommended for production)
- **Storage**: 20GB+ available space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Software Requirements
- **Python**: 3.8+ (3.11+ recommended)
- **PostgreSQL**: 12+ (15+ recommended)
- **Redis**: 6+ (7+ recommended)
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for multi-container deployment)

## 🛠️ Development Setup

### 1. Local Development

```bash
# Clone the repository
git clone <repository-url>
cd email-sequence-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-fastapi.txt

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Setup services
python start.py setup

# Run migrations
python start.py migrate

# Start development server
python start.py dev
```

### 2. Docker Development

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f email-sequence-api

# Stop services
docker-compose down
```

## 🏭 Production Deployment

### Option 1: Docker Compose (Recommended)

#### 1. Prepare Environment
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with production values

# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### 2. Deploy with Docker Compose
```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f email-sequence-api

# Run health check
docker-compose exec email-sequence-api python start.py health
```

#### 3. SSL/TLS Configuration
```bash
# Generate SSL certificates (Let's Encrypt)
certbot certonly --standalone -d yourdomain.com

# Copy certificates to ssl directory
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./ssl/
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./ssl/

# Restart nginx
docker-compose restart nginx
```

### Option 2: Manual Deployment

#### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip postgresql redis-server nginx

# Create application user
sudo useradd -m -s /bin/bash emailsequence
sudo usermod -aG sudo emailsequence
```

#### 2. Database Setup
```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE email_sequences;
CREATE USER emailsequence WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE email_sequences TO emailsequence;
\q
```

#### 3. Application Deployment
```bash
# Switch to application user
sudo su - emailsequence

# Clone repository
git clone <repository-url> /home/emailsequence/email-sequence-ai
cd /home/emailsequence/email-sequence-ai

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-fastapi.txt

# Configure environment
cp .env.example .env
# Edit .env with production values

# Setup services
python start.py setup

# Run migrations
python start.py migrate
```

#### 4. Systemd Service
```bash
# Create systemd service file
sudo nano /etc/systemd/system/email-sequence.service
```

```ini
[Unit]
Description=Email Sequence AI API
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=emailsequence
Group=emailsequence
WorkingDirectory=/home/emailsequence/email-sequence-ai
Environment=PATH=/home/emailsequence/email-sequence-ai/venv/bin
ExecStart=/home/emailsequence/email-sequence-ai/venv/bin/python start.py prod
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable email-sequence
sudo systemctl start email-sequence

# Check status
sudo systemctl status email-sequence
```

#### 5. Nginx Configuration
```bash
# Create nginx configuration
sudo nano /etc/nginx/sites-available/email-sequence
```

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # API proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Static files (if any)
    location /static/ {
        alias /home/emailsequence/email-sequence-ai/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/email-sequence /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Option 3: Kubernetes Deployment

#### 1. Create Namespace
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: email-sequence
```

#### 2. ConfigMap
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: email-sequence-config
  namespace: email-sequence
data:
  DATABASE_URL: "postgresql+asyncpg://postgres:password@postgres:5432/email_sequences"
  REDIS_URL: "redis://redis:6379/0"
  ENVIRONMENT: "production"
  DEBUG: "false"
```

#### 3. Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: email-sequence-api
  namespace: email-sequence
spec:
  replicas: 3
  selector:
    matchLabels:
      app: email-sequence-api
  template:
    metadata:
      labels:
        app: email-sequence-api
    spec:
      containers:
      - name: email-sequence-api
        image: your-registry/email-sequence-ai:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: email-sequence-config
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
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

#### 4. Service
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: email-sequence-service
  namespace: email-sequence
spec:
  selector:
    app: email-sequence-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

## 📊 Monitoring Setup

### 1. Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'email-sequence-api'
    static_configs:
      - targets: ['email-sequence-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### 2. Grafana Dashboard
Import the provided dashboard JSON file or create custom dashboards for:
- API performance metrics
- Email delivery statistics
- Database performance
- Cache hit rates
- Error rates and types

## 🔧 Maintenance

### 1. Database Backups
```bash
# Create backup script
#!/bin/bash
BACKUP_DIR="/backups/email-sequences"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U postgres email_sequences > "$BACKUP_DIR/backup_$DATE.sql"

# Schedule with cron
0 2 * * * /path/to/backup_script.sh
```

### 2. Log Rotation
```bash
# Configure logrotate
sudo nano /etc/logrotate.d/email-sequence
```

```
/home/emailsequence/email-sequence-ai/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 emailsequence emailsequence
    postrotate
        systemctl reload email-sequence
    endscript
}
```

### 3. Updates and Deployments
```bash
# Update application
cd /home/emailsequence/email-sequence-ai
git pull origin main
source venv/bin/activate
pip install -r requirements-fastapi.txt
python start.py migrate
sudo systemctl restart email-sequence

# Rollback if needed
git checkout <previous-commit>
sudo systemctl restart email-sequence
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database status
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT 1;"

# Check connection from application
python start.py health
```

#### 2. Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis
redis-cli ping

# Check Redis logs
sudo journalctl -u redis -f
```

#### 3. High Memory Usage
```bash
# Monitor memory usage
htop
free -h

# Check application logs
sudo journalctl -u email-sequence -f
```

#### 4. Slow Performance
```bash
# Check database performance
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"

# Check Redis performance
redis-cli --latency-history

# Check application metrics
curl http://localhost:9090/metrics
```

## 📈 Scaling

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Deploy multiple application instances
- Use Redis Cluster for caching
- Use PostgreSQL read replicas

### Vertical Scaling
- Increase server resources
- Optimize database queries
- Implement connection pooling
- Use SSD storage

## 🔒 Security

### 1. Firewall Configuration
```bash
# UFW configuration
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. SSL/TLS
- Use Let's Encrypt for free SSL certificates
- Configure HSTS headers
- Use strong cipher suites
- Regular certificate renewal

### 3. Database Security
- Use strong passwords
- Limit database access
- Enable SSL connections
- Regular security updates

## 📞 Support

For deployment issues:
1. Check the logs: `docker-compose logs -f email-sequence-api`
2. Run health check: `python start.py health`
3. Review configuration: `python start.py config`
4. Check system resources: `htop`, `df -h`, `free -h`

For additional support, create an issue on GitHub or contact the development team.






























