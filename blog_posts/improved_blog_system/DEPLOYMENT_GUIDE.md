# 🚀 Deployment Guide - Improved Blog System

This guide provides comprehensive instructions for deploying the improved blog system to production environments.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **PostgreSQL**: 13 or higher
- **Redis**: 6 or higher
- **Memory**: Minimum 2GB RAM
- **Storage**: Minimum 10GB free space
- **CPU**: 2 cores minimum

### Software Dependencies
- Docker and Docker Compose (recommended)
- Git
- SSL certificate (for HTTPS)

## 🐳 Docker Deployment (Recommended)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd improved_blog_system

# Copy environment configuration
cp .env.example .env
# Edit .env with your production settings
```

### 2. Configure Environment Variables

```bash
# .env file for production
DATABASE_URL=postgresql+asyncpg://postgres:your_secure_password@db/blog_db
REDIS_URL=redis://redis:6379
SECRET_KEY=your-super-secure-secret-key-here
DEBUG=false
API_TITLE=Blog System API
API_VERSION=1.0.0
CORS_ORIGINS=["https://yourdomain.com"]
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

### 3. Deploy with Docker Compose

```bash
# Build and start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f app
```

### 4. Run Database Migrations

```bash
# Execute migrations
docker-compose exec app alembic upgrade head

# Create initial admin user (optional)
docker-compose exec app python scripts/create_admin.py
```

## 🖥️ Manual Deployment

### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Install Redis
sudo apt install redis-server -y

# Install Nginx
sudo apt install nginx -y
```

### 2. Database Setup

```bash
# Create database and user
sudo -u postgres psql
CREATE DATABASE blog_db;
CREATE USER blog_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE blog_db TO blog_user;
\q
```

### 3. Application Setup

```bash
# Create application directory
sudo mkdir -p /opt/blog-system
sudo chown $USER:$USER /opt/blog-system
cd /opt/blog-system

# Clone repository
git clone <repository-url> .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with production settings

# Run migrations
alembic upgrade head
```

### 4. Systemd Service

Create `/etc/systemd/system/blog-system.service`:

```ini
[Unit]
Description=Blog System API
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/blog-system
Environment=PATH=/opt/blog-system/venv/bin
ExecStart=/opt/blog-system/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable blog-system
sudo systemctl start blog-system
sudo systemctl status blog-system
```

## 🌐 Nginx Configuration

### 1. Create Nginx Configuration

Create `/etc/nginx/sites-available/blog-system`:

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Proxy to FastAPI application
    location / {
        proxy_pass http://127.0.0.1:8000;
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
    
    # Static files
    location /static/ {
        alias /opt/blog-system/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # File uploads
    location /files/ {
        alias /opt/blog-system/uploads/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
```

### 2. Enable Site

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/blog-system /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

## 🔒 SSL Certificate Setup

### Using Let's Encrypt (Recommended)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring and Logging

### 1. Log Management

```bash
# Create log directory
sudo mkdir -p /var/log/blog-system
sudo chown www-data:www-data /var/log/blog-system

# Configure logrotate
sudo nano /etc/logrotate.d/blog-system
```

Logrotate configuration:
```
/var/log/blog-system/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload blog-system
    endscript
}
```

### 2. Monitoring Setup

```bash
# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Create monitoring script
sudo nano /opt/blog-system/scripts/monitor.sh
```

Monitoring script:
```bash
#!/bin/bash
# Check service status
systemctl is-active --quiet blog-system || echo "Blog system is down!"

# Check database connection
pg_isready -h localhost -p 5432 || echo "Database is down!"

# Check Redis connection
redis-cli ping || echo "Redis is down!"

# Check disk space
df -h | awk '$5 > 80 {print $0}'

# Check memory usage
free -h | awk 'NR==2{printf "Memory Usage: %s/%s (%.2f%%)\n", $3,$2,$3*100/$2 }'
```

## 🔄 Backup Strategy

### 1. Database Backup

```bash
# Create backup script
sudo nano /opt/blog-system/scripts/backup.sh
```

Backup script:
```bash
#!/bin/bash
BACKUP_DIR="/opt/backups/blog-system"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h localhost -U blog_user blog_db > $BACKUP_DIR/db_backup_$DATE.sql

# File uploads backup
tar -czf $BACKUP_DIR/uploads_backup_$DATE.tar.gz /opt/blog-system/uploads/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 2. Automated Backups

```bash
# Make script executable
sudo chmod +x /opt/blog-system/scripts/backup.sh

# Add to crontab
sudo crontab -e
# Add: 0 2 * * * /opt/blog-system/scripts/backup.sh
```

## 🚀 Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_blog_posts_status_created ON blog_posts(status, created_at);
CREATE INDEX CONCURRENTLY idx_blog_posts_author_status ON blog_posts(author_id, status);
CREATE INDEX CONCURRENTLY idx_comments_post_created ON comments(post_id, created_at);
CREATE INDEX CONCURRENTLY idx_likes_post_user ON likes(post_id, user_id);

-- Analyze tables
ANALYZE blog_posts;
ANALYZE comments;
ANALYZE likes;
```

### 2. Redis Configuration

```bash
# Edit Redis configuration
sudo nano /etc/redis/redis.conf
```

Key settings:
```
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 3. Application Optimization

```bash
# Set environment variables for production
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Use production WSGI server
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔧 Maintenance

### 1. Regular Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python dependencies
cd /opt/blog-system
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart services
sudo systemctl restart blog-system
sudo systemctl restart nginx
```

### 2. Health Checks

```bash
# Create health check script
sudo nano /opt/blog-system/scripts/health_check.sh
```

Health check script:
```bash
#!/bin/bash
# Check API health
curl -f http://localhost:8000/health || exit 1

# Check database
pg_isready -h localhost -p 5432 || exit 1

# Check Redis
redis-cli ping || exit 1

echo "All services are healthy"
```

## 🚨 Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   sudo journalctl -u blog-system -f
   
   # Check configuration
   sudo systemctl status blog-system
   ```

2. **Database connection issues**
   ```bash
   # Check PostgreSQL status
   sudo systemctl status postgresql
   
   # Test connection
   psql -h localhost -U blog_user -d blog_db
   ```

3. **High memory usage**
   ```bash
   # Check memory usage
   free -h
   
   # Check processes
   ps aux --sort=-%mem | head
   ```

4. **SSL certificate issues**
   ```bash
   # Check certificate
   sudo certbot certificates
   
   # Renew certificate
   sudo certbot renew
   ```

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancer Setup**
   - Use Nginx or HAProxy
   - Configure multiple application instances
   - Implement session affinity if needed

2. **Database Scaling**
   - Set up read replicas
   - Implement connection pooling
   - Consider database sharding for large datasets

3. **Cache Scaling**
   - Use Redis Cluster
   - Implement cache warming strategies
   - Monitor cache hit rates

## 🔐 Security Checklist

- [ ] SSL/TLS certificates installed and configured
- [ ] Firewall rules configured
- [ ] Database access restricted
- [ ] Strong passwords and secrets
- [ ] Regular security updates
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting set up
- [ ] Access logs enabled
- [ ] Rate limiting configured
- [ ] Security headers implemented

## 📞 Support

For deployment issues or questions:

1. Check the logs: `sudo journalctl -u blog-system -f`
2. Verify configuration: `sudo nginx -t`
3. Test database connection: `pg_isready`
4. Check service status: `sudo systemctl status blog-system`

This deployment guide provides a solid foundation for running the improved blog system in production. Adjust configurations based on your specific requirements and infrastructure.






























