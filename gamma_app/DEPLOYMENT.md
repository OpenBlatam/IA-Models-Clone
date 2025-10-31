# Gamma App - Deployment Guide

## 🚀 Production Deployment

### Prerequisites

- Docker and Docker Compose
- PostgreSQL 14+
- Redis 6+
- Python 3.8+
- 4GB+ RAM
- 20GB+ disk space

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd gamma_app
   cp env.example .env
   # Edit .env with your configuration
   ```

2. **Run Setup Script**
   ```bash
   python scripts/setup.py
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

4. **Verify Deployment**
   ```bash
   curl http://localhost:8000/health
   ```

### Environment Configuration

#### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/gamma_db

# Redis
REDIS_URL=redis://localhost:6379

# AI APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your_secret_key

# Email (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
```

### Docker Deployment

#### Single Container
```bash
docker build -t gamma-app .
docker run -d -p 8000:8000 --env-file .env gamma-app
```

#### Multi-Container (Recommended)
```bash
docker-compose -f docker-compose.yml up -d
```

### Kubernetes Deployment

#### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: gamma-app
```

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gamma-app-config
  namespace: gamma-app
data:
  config.yaml: |
    app:
      name: "Gamma App"
      environment: "production"
    database:
      url: "postgresql://user:pass@postgres:5432/gamma_db"
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gamma-app
  namespace: gamma-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gamma-app
  template:
    metadata:
      labels:
        app: gamma-app
    spec:
      containers:
      - name: gamma-app
        image: gamma-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gamma-app-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Scaling

#### Horizontal Scaling
```bash
# Scale API instances
kubectl scale deployment gamma-app --replicas=5

# Scale with Docker Compose
docker-compose up -d --scale gamma_app=3
```

#### Vertical Scaling
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Monitoring

#### Prometheus Metrics
- Endpoint: `http://localhost:9090/metrics`
- Key metrics: response_time, error_rate, throughput

#### Health Checks
- Liveness: `GET /health/live`
- Readiness: `GET /health/ready`
- Detailed: `GET /health/detailed`

#### Logging
- Structured JSON logs
- Log level: INFO (production)
- Log rotation: 100MB files, 5 backups

### Security

#### SSL/TLS
```nginx
server {
    listen 443 ssl;
    server_name gamma.app;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://gamma-app:8000;
    }
}
```

#### Firewall Rules
```bash
# Allow HTTP/HTTPS
ufw allow 80
ufw allow 443

# Allow SSH
ufw allow 22

# Allow internal services
ufw allow from 10.0.0.0/8 to any port 5432
ufw allow from 10.0.0.0/8 to any port 6379
```

### Backup and Recovery

#### Database Backup
```bash
# Create backup
pg_dump gamma_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
psql gamma_db < backup_20240101_120000.sql
```

#### Application Backup
```bash
# Backup uploads
tar -czf uploads_backup.tar.gz uploads/

# Backup configuration
tar -czf config_backup.tar.gz config/
```

### Troubleshooting

#### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database status
   docker-compose logs postgres
   
   # Test connection
   python -c "import psycopg2; psycopg2.connect('postgresql://user:pass@localhost:5432/gamma_db')"
   ```

2. **Redis Connection Failed**
   ```bash
   # Check Redis status
   docker-compose logs redis
   
   # Test connection
   redis-cli ping
   ```

3. **AI API Errors**
   ```bash
   # Check API keys
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   
   # Test API access
   curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
   ```

#### Performance Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Optimize cache settings
   # Reduce model cache size
   ```

2. **Slow Response Times**
   ```bash
   # Check database performance
   # Optimize queries
   # Increase cache TTL
   ```

### Maintenance

#### Regular Tasks

1. **Daily**
   - Check health status
   - Monitor error rates
   - Review logs

2. **Weekly**
   - Update dependencies
   - Run security scans
   - Backup data

3. **Monthly**
   - Performance review
   - Capacity planning
   - Security audit

#### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Run migrations
python scripts/migrate.py
```

### Support

- Documentation: [README.md](README.md)
- API Docs: http://localhost:8000/docs
- Issues: GitHub Issues
- Monitoring: http://localhost:3000 (Grafana)

























