# Suno Clone AI - EC2 Deployment Guide

Complete guide for deploying Suno Clone AI to AWS EC2 with automated CI/CD pipeline.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Manual Deployment](#manual-deployment)
- [Ansible Deployment](#ansible-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## 🎯 Overview

This deployment guide provides multiple methods to deploy Suno Clone AI to AWS EC2:

1. **Automated CI/CD** - GitHub Actions workflow (recommended)
2. **Ansible Playbook** - Infrastructure-as-Code approach
3. **Bash Script** - Manual deployment script

## 📦 Prerequisites

### Required Tools

- **AWS Account** with EC2 instance running
- **Docker** installed on EC2 instance
- **Git** for version control
- **SSH access** to EC2 instance
- **Python 3.11+** (for local development)

### EC2 Instance Requirements

- **OS**: Ubuntu 20.04+ or Amazon Linux 2
- **Instance Type**: t3.medium or larger (recommended: t3.large for GPU support)
- **Storage**: Minimum 20GB free space
- **Security Group**: Allow inbound traffic on port 8020
- **IAM Role**: EC2 instance should have appropriate permissions for AWS services (if using S3, DynamoDB, etc.)

### GitHub Secrets Configuration

Configure the following secrets in your GitHub repository:

```
EC2_HOST=your-ec2-instance-ip-or-domain
EC2_USER=ubuntu
EC2_SSH_KEY=your-private-ssh-key-content
SLACK_WEBHOOK_URL=your-slack-webhook-url (optional)
```

To add secrets:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret above

## 🏗️ Architecture

```
┌─────────────────┐
│  GitHub Actions  │
│   (CI/CD)        │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Build & Test   │
│  Docker Image   │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Deploy to EC2  │
│  via SSH/SCP    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  EC2 Instance   │
│  ┌───────────┐  │
│  │  Docker   │  │
│  │ Container │  │
│  │ Port 8020 │  │
│  └───────────┘  │
└─────────────────┘
```

## 🚀 Quick Start

### Automated Deployment (Recommended)

1. **Configure GitHub Secrets** (see Prerequisites)

2. **Push to main branch**:
   ```bash
   git add .
   git commit -m "Deploy Suno Clone AI"
   git push origin main
   ```

3. **Monitor deployment**:
   - Go to GitHub Actions tab
   - Watch the deployment workflow
   - Check logs for any errors

The workflow will automatically:
- Run linting and tests
- Build Docker image
- Deploy to EC2
- Run health checks
- Send notifications

## 🔄 GitHub Actions CI/CD

### Workflow Overview

The GitHub Actions workflow (`.github/workflows/deploy-suno-clone-ai.yml`) includes:

1. **Lint and Test Job**:
   - Code linting with flake8
   - Format checking with black
   - Type checking with mypy
   - Unit tests with pytest
   - Coverage reporting

2. **Build and Deploy Job**:
   - Docker image build
   - Image compression and transfer
   - EC2 deployment
   - Health checks
   - Notifications

### Workflow Triggers

- **Automatic**: Push to `main` branch (only for `suno_clone_ai` directory)
- **Manual**: Workflow dispatch with environment selection

### Customization

Edit `.github/workflows/deploy-suno-clone-ai.yml` to customize:
- Deployment environment
- Build steps
- Notification channels
- Health check intervals

## 🛠️ Manual Deployment

### Using Bash Script

1. **Prepare deployment files**:
   ```bash
   cd agents/backend/onyx/server/features/suno_clone_ai
   
   # Build Docker image
   docker build -t suno-clone-ai:latest .
   
   # Save image
   docker save suno-clone-ai:latest | gzip > suno-clone-ai.tar.gz
   ```

2. **Transfer files to EC2**:
   ```bash
   scp -i ~/.ssh/your-key.pem suno-clone-ai.tar.gz ubuntu@your-ec2-host:~/suno-clone-ai/
   scp -i ~/.ssh/your-key.pem .env ubuntu@your-ec2-host:~/suno-clone-ai/
   scp -i ~/.ssh/your-key.pem deploy/deploy.sh ubuntu@your-ec2-host:~/suno-clone-ai/
   ```

3. **Run deployment script**:
   ```bash
   ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-host
   cd ~/suno-clone-ai
   chmod +x deploy.sh
   ./deploy.sh
   ```

### Using Docker Compose (Alternative)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  suno-clone-ai:
    image: suno-clone-ai:latest
    container_name: suno-clone-ai
    restart: unless-stopped
    ports:
      - "8020:8020"
    env_file:
      - .env
    volumes:
      - ./storage:/app/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8020/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Deploy:
```bash
docker-compose up -d
```

## 🤖 Ansible Deployment

### Prerequisites

```bash
pip install ansible ansible-lint
```

### Configuration

1. **Update inventory** (`deploy/ansible/inventory.yml`):
   ```yaml
   suno-clone-ai-prod:
     ansible_host: your-ec2-host.com
     ansible_user: ubuntu
     ansible_ssh_private_key_file: ~/.ssh/your-key.pem
   ```

2. **Encrypt sensitive variables** (optional):
   ```bash
   ansible-vault encrypt deploy/ansible/group_vars/all.yml
   ```

### Deployment

```bash
cd deploy/ansible

# Dry run
ansible-playbook playbook.yml --check

# Deploy
ansible-playbook playbook.yml

# Deploy with vault password
ansible-playbook playbook.yml --ask-vault-pass
```

### Ansible Tags

Deploy specific components:
```bash
# Only update Docker
ansible-playbook playbook.yml --tags docker

# Only deploy application
ansible-playbook playbook.yml --tags deploy
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file on EC2:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8020
DEBUG=False

# OpenAI (optional)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# Music Generation
MUSIC_MODEL=facebook/musicgen-medium
USE_GPU=True
MAX_AUDIO_LENGTH=300
DEFAULT_DURATION=30
SAMPLE_RATE=32000

# Storage
AUDIO_STORAGE_PATH=./storage/audio
DATABASE_URL=sqlite:///./suno_clone.db

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key

# AWS (if using)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=your-bucket-name
```

### Security Best Practices

1. **Never commit `.env` files** to Git
2. **Use AWS Secrets Manager** or Parameter Store for production
3. **Rotate secrets** regularly
4. **Use IAM roles** instead of access keys when possible
5. **Enable SSL/TLS** for production deployments

## 🔍 Troubleshooting

### Common Issues

#### 1. Docker Image Not Loading

**Error**: `Failed to load Docker image`

**Solution**:
```bash
# Check image file integrity
gunzip -t suno-clone-ai.tar.gz

# Try loading manually
docker load < suno-clone-ai.tar.gz
```

#### 2. Container Fails to Start

**Error**: Container exits immediately

**Solution**:
```bash
# Check logs
docker logs suno-clone-ai

# Check environment variables
docker exec suno-clone-ai env

# Verify health endpoint
curl http://localhost:8020/health
```

#### 3. Port Already in Use

**Error**: `Bind for 0.0.0.0:8020 failed: port is already allocated`

**Solution**:
```bash
# Find process using port
sudo lsof -i :8020

# Stop existing container
docker stop suno-clone-ai
docker rm suno-clone-ai
```

#### 4. Health Check Fails

**Error**: Health check times out

**Solution**:
- Check application logs: `docker logs suno-clone-ai`
- Verify health endpoint: `curl http://localhost:8020/health`
- Increase health check start period in Dockerfile

#### 5. Out of Memory

**Error**: Container killed due to OOM

**Solution**:
- Upgrade EC2 instance type
- Reduce model size (use `musicgen-small` instead of `medium`)
- Enable model quantization

### Debugging Commands

```bash
# Check container status
docker ps -a | grep suno-clone-ai

# View logs
docker logs suno-clone-ai --tail 100 -f

# Execute commands in container
docker exec -it suno-clone-ai bash

# Check resource usage
docker stats suno-clone-ai

# Inspect container
docker inspect suno-clone-ai
```

## ✅ Best Practices

### Security

1. **Use secrets management** (AWS Secrets Manager, HashiCorp Vault)
2. **Enable firewall** (UFW, iptables)
3. **Use HTTPS** with reverse proxy (Nginx, Traefik)
4. **Regular security updates**
5. **Least privilege** IAM roles

### Monitoring

1. **CloudWatch Logs** for application logs
2. **CloudWatch Metrics** for performance
3. **Health checks** with alerts
4. **Uptime monitoring** (Pingdom, UptimeRobot)

### Backup

1. **Database backups** (automated daily)
2. **Audio storage backups** to S3
3. **Configuration backups**
4. **Disaster recovery plan**

### Performance

1. **Use GPU instances** for faster generation
2. **Enable caching** (Redis, ElastiCache)
3. **CDN** for static assets
4. **Load balancing** for high availability

### Maintenance

1. **Regular updates** (security patches)
2. **Monitor disk space**
3. **Rotate logs**
4. **Review and optimize** resource usage

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Ansible Documentation](https://docs.ansible.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)

## 🤝 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review application logs
3. Open an issue on GitHub
4. Contact the development team

---

**Last Updated**: 2024
**Version**: 1.0.0




