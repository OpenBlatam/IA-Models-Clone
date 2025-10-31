# Blaze AI Production Deployment Guide

This guide provides comprehensive instructions for deploying the Blaze AI platform in production environments using Docker, Kubernetes, and best practices.

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (for containerized deployment)
- **Kubernetes** (for orchestrated deployment)
- **PostgreSQL 15+** (database)
- **Redis 7+** (caching)
- **Nginx** (reverse proxy)
- **SSL Certificates** (Let's Encrypt recommended)

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Database Configuration
DB_PASSWORD=your_secure_password_here
REDIS_PASSWORD=your_redis_password_here

# API Configuration
BLAZE_AI_API_KEY=your_api_key_here
BLAZE_AI_JWT_SECRET=your_jwt_secret_here

# SSL Configuration
SSL_CERT_PATH=./deployment/nginx/ssl/cert.pem
SSL_KEY_PATH=./deployment/nginx/ssl/key.pem
```

## 🐳 Docker Deployment

### 1. Build and Deploy

```bash
# Make deployment script executable
chmod +x deployment/scripts/deploy.sh

# Run deployment script
./deployment/scripts/deploy.sh

# Choose option 2 for Docker Compose
```

### 2. Manual Docker Compose

```bash
# Build and start services
docker-compose -f deployment/docker/docker-compose.prod.yml up -d --build

# Check status
docker-compose -f deployment/docker/docker-compose.prod.yml ps

# View logs
docker-compose -f deployment/docker/docker-compose.prod.yml logs -f
```

### 3. SSL Setup

```bash
# Make SSL script executable
chmod +x deployment/scripts/ssl-setup.sh

# Run SSL setup
./deployment/scripts/ssl-setup.sh

# Choose Let's Encrypt for production
```

## ☸️ Kubernetes Deployment

### 1. Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- nginx-ingress controller
- cert-manager (for SSL)

### 2. Deploy

```bash
# Make deployment script executable
chmod +x deployment/scripts/deploy.sh

# Run deployment script
./deployment/scripts/deploy.sh

# Choose option 1 for Kubernetes
```

### 3. Manual Kubernetes Deployment

```bash
# Create namespace and RBAC
kubectl apply -f deployment/kubernetes/namespace.yaml

# Create storage
kubectl apply -f deployment/kubernetes/storage.yaml

# Create secrets
kubectl apply -f deployment/kubernetes/secrets.yaml

# Create monitoring
kubectl apply -f deployment/kubernetes/monitoring.yaml

# Create deployment
kubectl apply -f deployment/kubernetes/deployment.yaml

# Create ingress
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### 4. Check Status

```bash
# Check pods
kubectl get pods -n blaze-ai

# Check services
kubectl get services -n blaze-ai

# Check ingress
kubectl get ingress -n blaze-ai

# Check persistent volumes
kubectl get pvc -n blaze-ai
```

## 🔒 Security Configuration

### 1. SSL/TLS Setup

```bash
# Generate Let's Encrypt certificate
./deployment/scripts/ssl-setup.sh

# Verify certificate
openssl x509 -in deployment/nginx/ssl/cert.pem -text -noout
```

### 2. Firewall Configuration

```bash
# Allow required ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 8001/tcp
sudo ufw allow 8002/tcp

# Enable firewall
sudo ufw enable
```

### 3. Security Headers

The nginx configuration includes:
- X-Frame-Options
- X-Content-Type-Options
- X-XSS-Protection
- Referrer-Policy
- Content-Security-Policy

## 📊 Monitoring and Observability

### 1. Prometheus Metrics

- **Endpoint**: `/metrics`
- **Port**: 8002
- **Scraping**: Every 15 seconds

### 2. Grafana Dashboards

- **URL**: http://localhost:3000
- **Default**: admin/admin-password
- **Port**: 3000

### 3. Health Checks

```bash
# API Health
curl http://localhost:8000/health

# Gradio Health
curl http://localhost:8001/health

# Metrics Health
curl http://localhost:8002/health
```

## 💾 Backup and Recovery

### 1. Automated Backups

```bash
# Make backup script executable
chmod +x deployment/scripts/backup.sh

# Run backup
./deployment/scripts/backup.sh

# Setup cron job for daily backups
crontab -e
# Add: 0 2 * * * /path/to/blaze-ai/deployment/scripts/backup.sh
```

### 2. Backup Components

- PostgreSQL database
- Redis cache
- Application files
- AI models
- Application logs

### 3. Restore Process

```bash
# Database restore
PGPASSWORD=$DB_PASSWORD psql -h localhost -U blazeai -d blazeai < backup_file.sql

# File restore
tar -xzf backup_file.tar

# Redis restore
redis-cli -a $REDIS_PASSWORD --rdb backup_file.rdb
```

## 🔧 Maintenance and Updates

### 1. Rolling Updates

```bash
# Kubernetes rolling update
kubectl rollout restart deployment/blaze-ai -n blaze-ai

# Docker Compose update
docker-compose -f deployment/docker/docker-compose.prod.yml pull
docker-compose -f deployment/docker/docker-compose.prod.yml up -d
```

### 2. Log Management

```bash
# View application logs
docker-compose -f deployment/docker/docker-compose.prod.yml logs -f blaze-ai

# Kubernetes logs
kubectl logs -f deployment/blaze-ai -n blaze-ai

# Log rotation
sudo logrotate /etc/logrotate.d/blaze-ai
```

### 3. Performance Tuning

```bash
# Check resource usage
docker stats
kubectl top pods -n blaze-ai

# Monitor memory and CPU
htop
iotop
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   sudo netstat -tulpn | grep :8000
   
   # Kill process using port
   sudo kill -9 <PID>
   ```

2. **SSL Certificate Issues**
   ```bash
   # Check certificate validity
   openssl x509 -in cert.pem -text -noout
   
   # Renew Let's Encrypt
   certbot renew
   ```

3. **Database Connection Issues**
   ```bash
   # Test PostgreSQL connection
   PGPASSWORD=$DB_PASSWORD psql -h localhost -U blazeai -d blazeai -c "SELECT 1;"
   
   # Check Redis connection
   redis-cli -a $REDIS_PASSWORD ping
   ```

4. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   
   # Clear Docker cache
   docker system prune -a
   ```

### Debug Mode

```bash
# Enable debug logging
export BLAZE_AI_LOG_LEVEL=DEBUG

# Run with debug
docker-compose -f deployment/docker/docker-compose.prod.yml up --build
```

## 📈 Scaling

### 1. Horizontal Pod Autoscaler (Kubernetes)

```bash
# Check HPA status
kubectl get hpa -n blaze-ai

# Scale manually
kubectl scale deployment blaze-ai --replicas=5 -n blaze-ai
```

### 2. Docker Compose Scaling

```bash
# Scale services
docker-compose -f deployment/docker/docker-compose.prod.yml up -d --scale blaze-ai=3
```

### 3. Load Balancer Configuration

The nginx configuration includes:
- Least connections load balancing
- Health checks
- Rate limiting
- Connection pooling

## 🔄 CI/CD Integration

### 1. GitHub Actions

```yaml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/blaze-ai blaze-ai=blaze-ai:latest -n blaze-ai
```

### 2. Docker Registry

```bash
# Build and push image
docker build -t your-registry/blaze-ai:latest .
docker push your-registry/blaze-ai:latest

# Update deployment
kubectl set image deployment/blaze-ai blaze-ai=your-registry/blaze-ai:latest -n blaze-ai
```

## 📋 Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Database backups scheduled
- [ ] Monitoring configured
- [ ] Log rotation enabled
- [ ] Health checks working
- [ ] Rate limiting configured
- [ ] Security headers enabled
- [ ] Backup restoration tested

## 🆘 Support

For production deployment issues:

1. Check the logs: `docker-compose logs` or `kubectl logs`
2. Verify configuration files
3. Check system resources
4. Review security settings
5. Test backup/restore procedures

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
