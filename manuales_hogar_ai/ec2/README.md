# EC2 Deployment Guide - Manuales Hogar AI

## 🚀 Quick Start

### Option 1: Using User Data (Automatic Setup)

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS or Amazon Linux 2023
   - Instance Type: t3.medium or larger (recommended: t3.large)
   - Storage: 20GB minimum
   - Security Group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

2. **Configure User Data**
   - Copy contents of `user-data.sh` to EC2 User Data
   - Or use the script directly in Launch Template

3. **After Instance Launch**
   ```bash
   # SSH to instance
   ssh ubuntu@your-ec2-ip

   # Copy application files
   sudo mkdir -p /opt/manuales-hogar-ai
   # Upload your application files here

   # Configure environment
   sudo nano /opt/manuales-hogar-ai/.env

   # Start service
   sudo systemctl start manuales-hogar-ai
   ```

### Option 2: Manual Setup

```bash
# On your local machine
export EC2_HOST=your-ec2-ip-or-hostname
export EC2_USER=ubuntu

# Run deployment script
./ec2/deploy.sh
```

### Option 3: Manual Installation

```bash
# SSH to EC2 instance
ssh ubuntu@your-ec2-ip

# Clone or upload application
cd /opt
sudo mkdir -p manuales-hogar-ai
cd manuales-hogar-ai
# Copy your application files here

# Run setup script
sudo bash ec2/setup.sh

# Configure environment
sudo nano .env

# Start service
sudo systemctl start manuales-hogar-ai
```

## 📋 Prerequisites

### EC2 Instance Requirements

- **OS**: Ubuntu 22.04 LTS or Amazon Linux 2023
- **Instance Type**: 
  - Minimum: t3.small (2 vCPU, 2GB RAM)
  - Recommended: t3.medium (2 vCPU, 4GB RAM) or larger
  - For production: t3.large (2 vCPU, 8GB RAM) or larger
- **Storage**: 20GB minimum (SSD recommended)
- **Network**: 
  - Security group with ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)
  - Public IP or Elastic IP

### Required Software (Installed Automatically)

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Nginx
- PostgreSQL client
- Redis tools
- AWS CLI

## ⚙️ Configuration

### Environment Variables

Edit `/opt/manuales-hogar-ai/.env`:

```bash
# Application
ENVIRONMENT=prod
DEBUG=false
PORT=8000
WORKERS=4

# Database (if using external RDS)
DB_HOST=your-rds-endpoint.rds.amazonaws.com
DB_PORT=5432
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=manuales_db
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Redis (if using external ElastiCache)
REDIS_URL=redis://your-elasticache-endpoint:6379/0

# OpenRouter API
OPENROUTER_API_KEY=your_api_key_here

# Security
ALLOWED_ORIGINS=https://yourdomain.com
SECRET_KEY=your_secret_key_here

# Monitoring
ENABLE_PROMETHEUS=true
ENABLE_TRACING=false
```

### Nginx Configuration

Edit `/etc/nginx/sites-available/manuales-hogar-ai`:

```nginx
server_name yourdomain.com;  # Replace with your domain
```

### SSL Certificate (Let's Encrypt)

```bash
# Install certbot (already installed by setup script)
sudo certbot --nginx -d yourdomain.com

# Auto-renewal is configured automatically
```

## 🔧 Service Management

### Start Service

```bash
sudo systemctl start manuales-hogar-ai
```

### Stop Service

```bash
sudo systemctl stop manuales-hogar-ai
```

### Restart Service

```bash
sudo systemctl restart manuales-hogar-ai
```

### Check Status

```bash
sudo systemctl status manuales-hogar-ai
```

### View Logs

```bash
# Systemd logs
sudo journalctl -u manuales-hogar-ai -f

# Application logs
docker logs -f manuales-hogar-ai-app-1

# Nginx logs
sudo tail -f /var/log/nginx/manuales-hogar-ai-access.log
sudo tail -f /var/log/nginx/manuales-hogar-ai-error.log
```

## 🐳 Docker Management

### View Containers

```bash
cd /opt/manuales-hogar-ai
docker compose ps
```

### View Logs

```bash
cd /opt/manuales-hogar-ai
docker compose logs -f
```

### Restart Containers

```bash
cd /opt/manuales-hogar-ai
docker compose restart
```

### Update Application

```bash
cd /opt/manuales-hogar-ai
# Pull latest code
git pull  # or upload new files

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 🔍 Monitoring

### Health Check

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

# Disk usage
df -h

# Docker stats
docker stats
```

## 🔒 Security

### Firewall (UFW)

```bash
# Allow SSH, HTTP, HTTPS
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Security Groups

Ensure EC2 Security Group allows:
- Port 22 (SSH) from your IP only
- Port 80 (HTTP) from anywhere
- Port 443 (HTTPS) from anywhere

### Regular Updates

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Update Docker images
cd /opt/manuales-hogar-ai
docker compose pull
docker compose up -d
```

## 📊 Scaling

### Vertical Scaling (Larger Instance)

1. Stop application
2. Change instance type in EC2 console
3. Restart instance
4. Start application

### Horizontal Scaling (Multiple Instances)

1. Create AMI from current instance
2. Launch new instances from AMI
3. Configure load balancer (ALB)
4. Update DNS to point to load balancer

## 🚨 Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u manuales-hogar-ai -n 50

# Check Docker
sudo systemctl status docker
docker ps -a

# Check .env file
cat /opt/manuales-hogar-ai/.env
```

### Database Connection Issues

```bash
# Test database connection
psql -h $DB_HOST -U $DB_USER -d $DB_NAME

# Check database URL in .env
grep DATABASE_URL /opt/manuales-hogar-ai/.env
```

### High Memory Usage

```bash
# Check memory
free -h

# Check Docker memory
docker stats

# Restart containers
cd /opt/manuales-hogar-ai
docker compose restart
```

### Port Already in Use

```bash
# Check what's using port 8000
sudo lsof -i :8000

# Kill process if needed
sudo kill -9 <PID>
```

## 📝 Backup

### Database Backup

```bash
# If using local PostgreSQL
docker exec manuales-hogar-ai-db-1 pg_dump -U manuales_user manuales_db > backup.sql

# If using RDS
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > backup.sql
```

### Application Backup

```bash
# Backup application directory
sudo tar -czf manuales-hogar-ai-backup-$(date +%Y%m%d).tar.gz /opt/manuales-hogar-ai
```

## 🔄 Updates

### Update Application Code

```bash
cd /opt/manuales-hogar-ai
# Pull latest code or upload new files
git pull  # or rsync/scp new files

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Update System Packages

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo reboot  # If kernel updated
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Systemd Documentation](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)

## 🆘 Support

For issues or questions:
1. Check logs: `sudo journalctl -u manuales-hogar-ai -f`
2. Check application logs: `docker compose logs -f`
3. Review this documentation
4. Check [main README](../README.md)

---

**Last Updated**: 2024-01-XX




