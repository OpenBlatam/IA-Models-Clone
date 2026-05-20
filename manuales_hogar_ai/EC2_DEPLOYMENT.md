# EC2 Deployment Guide - Manuales Hogar AI

## 🎯 Overview

This guide explains how to deploy Manuales Hogar AI on any EC2 instance, regardless of instance type or size.

## 🚀 Quick Start

### Method 1: Automated Deployment (Recommended)

```bash
# From your local machine
export EC2_HOST=your-ec2-ip-or-hostname
export EC2_USER=ubuntu
./ec2/deploy.sh
```

### Method 2: Using EC2 User Data

1. Launch EC2 instance
2. In "Advanced details" → "User data", paste contents of `ec2/user-data.sh`
3. Instance will auto-configure on first boot

### Method 3: Manual Setup

```bash
# SSH to EC2
ssh ubuntu@your-ec2-ip

# Clone or upload application
sudo mkdir -p /opt/manuales-hogar-ai
cd /opt/manuales-hogar-ai
# Upload your files here

# Run setup
sudo bash ec2/setup.sh

# Configure
sudo nano .env

# Start
sudo systemctl start manuales-hogar-ai
```

## 📋 Requirements

### EC2 Instance

- **OS**: Ubuntu 22.04 LTS (recommended) or Amazon Linux 2023
- **Instance Types**:
  - **Minimum**: t3.small (2 vCPU, 2GB RAM) - for testing
  - **Recommended**: t3.medium (2 vCPU, 4GB RAM) - for small production
  - **Production**: t3.large+ (2+ vCPU, 8GB+ RAM) - for production
- **Storage**: 20GB minimum (SSD recommended)
- **Network**: 
  - Security Group: Ports 22 (SSH), 80 (HTTP), 443 (HTTPS)
  - Public IP or Elastic IP

### What Gets Installed

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Nginx (reverse proxy)
- PostgreSQL client
- Redis tools
- Certbot (SSL certificates)
- AWS CLI

## ⚙️ Configuration

### 1. Environment Variables

Edit `/opt/manuales-hogar-ai/.env`:

```bash
# Application
ENVIRONMENT=prod
DEBUG=false
PORT=8000
WORKERS=4

# Database
# Option A: Use Docker PostgreSQL (default)
DB_HOST=localhost
DB_PORT=5432
DB_USER=manuales_user
DB_PASSWORD=your_password
DB_NAME=manuales_db

# Option B: Use external RDS
# DB_HOST=your-rds-endpoint.rds.amazonaws.com
# DATABASE_URL=postgresql://user:pass@host:5432/db

# Redis
# Option A: Use Docker Redis (default)
REDIS_URL=redis://localhost:6379/0

# Option B: Use external ElastiCache
# REDIS_URL=redis://your-elasticache-endpoint:6379/0

# OpenRouter API
OPENROUTER_API_KEY=your_api_key_here

# Security
ALLOWED_ORIGINS=https://yourdomain.com
SECRET_KEY=your_secret_key_here

# Monitoring
ENABLE_PROMETHEUS=true
ENABLE_TRACING=false
```

### 2. Nginx Configuration

Edit `/etc/nginx/sites-available/manuales-hogar-ai`:

```nginx
server_name yourdomain.com;  # Replace with your domain
```

### 3. SSL Certificate

```bash
sudo certbot --nginx -d yourdomain.com
```

## 🔧 Service Management

### Start/Stop/Restart

```bash
# Start
sudo systemctl start manuales-hogar-ai

# Stop
sudo systemctl stop manuales-hogar-ai

# Restart
sudo systemctl restart manuales-hogar-ai

# Status
sudo systemctl status manuales-hogar-ai

# Enable on boot
sudo systemctl enable manuales-hogar-ai
```

### View Logs

```bash
# Systemd logs
sudo journalctl -u manuales-hogar-ai -f

# Application logs
cd /opt/manuales-hogar-ai
docker compose logs -f

# Nginx logs
sudo tail -f /var/log/nginx/manuales-hogar-ai-access.log
sudo tail -f /var/log/nginx/manuales-hogar-ai-error.log
```

## 🐳 Docker Management

### View Status

```bash
cd /opt/manuales-hogar-ai
docker compose ps
```

### Restart Services

```bash
cd /opt/manuales-hogar-ai
docker compose restart
```

### Update Application

```bash
cd /opt/manuales-hogar-ai

# Pull latest code or upload new files
git pull  # or upload via rsync/scp

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 🔍 Health Checks

### Application Health

```bash
curl http://localhost:8000/api/v1/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### System Resources

```bash
# CPU and Memory
htop

# Docker stats
docker stats
```

## 🔒 Security

### Firewall Setup

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Security Groups

Ensure EC2 Security Group allows:
- **Port 22** (SSH): From your IP only
- **Port 80** (HTTP): From anywhere (0.0.0.0/0)
- **Port 443** (HTTPS): From anywhere (0.0.0.0/0)

### Regular Updates

```bash
# System updates
sudo apt-get update && sudo apt-get upgrade -y

# Application updates
cd /opt/manuales-hogar-ai
docker compose pull
docker compose up -d
```

## 📊 Scaling Options

### Vertical Scaling (Larger Instance)

1. Stop application: `sudo systemctl stop manuales-hogar-ai`
2. Create AMI snapshot
3. Change instance type in EC2 console
4. Restart instance
5. Start application: `sudo systemctl start manuales-hogar-ai`

### Horizontal Scaling (Multiple Instances)

1. Create AMI from current instance
2. Launch new instances from AMI
3. Configure Application Load Balancer (ALB)
4. Update DNS to point to ALB
5. Configure shared database (RDS) and cache (ElastiCache)

## 🚨 Troubleshooting

### Service Won't Start

```bash
# Check systemd logs
sudo journalctl -u manuales-hogar-ai -n 50

# Check Docker
sudo systemctl status docker
docker ps -a

# Check .env file
cat /opt/manuales-hogar-ai/.env
```

### Database Connection Issues

```bash
# Test connection
psql -h $DB_HOST -U $DB_USER -d $DB_NAME

# Check Docker database
cd /opt/manuales-hogar-ai
docker compose logs db
```

### High Memory Usage

```bash
# Check memory
free -h

# Restart containers
cd /opt/manuales-hogar-ai
docker compose restart
```

### Port Conflicts

```bash
# Check port usage
sudo lsof -i :8000

# Kill process if needed
sudo kill -9 <PID>
```

## 📝 Backup

### Database Backup

```bash
# Local PostgreSQL
docker exec manuales-hogar-ai-db-1 pg_dump -U manuales_user manuales_db > backup.sql

# RDS
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > backup.sql
```

### Application Backup

```bash
sudo tar -czf manuales-backup-$(date +%Y%m%d).tar.gz /opt/manuales-hogar-ai
```

## 🔄 Updates

### Update Application

```bash
cd /opt/manuales-hogar-ai

# Method 1: Git
git pull origin main

# Method 2: Upload new files
# Use rsync or scp to upload updated files

# Rebuild
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 📚 Instance Type Recommendations

### Development/Testing
- **t3.small**: 2 vCPU, 2GB RAM
- Cost: ~$15/month
- Suitable for: Testing, development

### Small Production
- **t3.medium**: 2 vCPU, 4GB RAM
- Cost: ~$30/month
- Suitable for: Small applications, <1000 users/day

### Medium Production
- **t3.large**: 2 vCPU, 8GB RAM
- Cost: ~$60/month
- Suitable for: Medium applications, <10,000 users/day

### Large Production
- **t3.xlarge**: 4 vCPU, 16GB RAM
- Cost: ~$120/month
- Suitable for: Large applications, <100,000 users/day

### High Performance
- **m5.large+**: 2+ vCPU, 8GB+ RAM
- Cost: Varies
- Suitable for: High-traffic applications

## 🎯 Best Practices

1. **Use Elastic IP**: Prevents IP changes on restart
2. **Enable CloudWatch**: Monitor instance metrics
3. **Regular Backups**: Automated database backups
4. **Security Updates**: Keep system packages updated
5. **SSL Certificates**: Use Let's Encrypt for HTTPS
6. **Log Rotation**: Configure log rotation for Nginx
7. **Resource Monitoring**: Monitor CPU, memory, disk usage

## 📖 Additional Resources

- [EC2 Setup Scripts](../ec2/)
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)

---

**Last Updated**: 2024-01-XX




