# Complete Setup Guide - AI Project Generator

Complete guide for setting up the AI Project Generator with AWS deployment and CI/CD.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [AWS Infrastructure Setup](#aws-infrastructure-setup)
4. [CI/CD Configuration](#cicd-configuration)
5. [Deployment](#deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Prerequisites

### Required Software

- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **Docker** and **Docker Compose**
- **Git**
- **AWS CLI** (for deployment)
- **Terraform** (>= 1.0)
- **Ansible** (>= 2.9)

### AWS Account

- AWS account with appropriate permissions
- IAM user with EC2, VPC, S3, ECR permissions
- SSH key pair created in AWS

## Local Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd ai_project_generator
```

### 2. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Development dependencies
pip install pre-commit pytest black ruff mypy

# Install pre-commit hooks
pre-commit install
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# Set REDIS_URL, MCP_SECRET_KEY, etc.
```

### 4. Run Locally

```bash
# Using Python
python main.py

# Using Docker
docker-compose up

# Using Make
make docker-run
```

## AWS Infrastructure Setup

### 1. Configure AWS CLI

```bash
aws configure
# Enter AWS Access Key ID
# Enter AWS Secret Access Key
# Enter default region (e.g., us-east-1)
```

### 2. Create SSH Key Pair

```bash
aws ec2 create-key-pair \
  --key-name ai-project-generator-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/ai-project-generator-key.pem
chmod 400 ~/.ssh/ai-project-generator-key.pem
```

### 3. Configure Terraform

```bash
cd aws/terraform
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars
# Set key_name, instance_type, etc.
```

### 4. Deploy Infrastructure

```bash
cd aws/terraform
terraform init
terraform plan
terraform apply
```

## CI/CD Configuration

### 1. Configure GitHub Secrets

Go to repository settings > Secrets and variables > Actions:

```
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
SLACK_WEBHOOK_URL=your-webhook-url (optional)
```

### 2. Enable Workflows

Workflows are automatically enabled when pushed to repository.

### 3. Test CI/CD Locally

```bash
# Run CI checks
make ci

# Run tests
make test

# Build Docker image
make build

# Validate deployment
cd scripts/ci_cd
./deploy.sh validate
```

## Deployment

### Automatic Deployment

1. **Push to develop** → Deploys to staging
2. **Merge to main** → Deploys to production
3. **Push tag** → Creates release

### Manual Deployment

```bash
# Deploy to staging
ENVIRONMENT=staging IMAGE_TAG=v1.0.0 \
  ./scripts/ci_cd/deploy.sh deploy

# Deploy to production
ENVIRONMENT=production IMAGE_TAG=v1.0.0 \
  AUTO_APPROVE=true \
  ./scripts/ci_cd/deploy.sh deploy
```

### Using Make

```bash
# Build
make build

# Deploy
ENVIRONMENT=staging make deploy

# Full CI/CD
make ci cd
```

## Monitoring and Maintenance

### Automated Scripts

```bash
# Setup cron jobs
sudo ./aws/scripts/setup_cron_jobs.sh

# Manual backup
./aws/scripts/automated_backup.sh

# Manual monitoring
./aws/scripts/automated_monitoring.sh

# Security audit
./aws/scripts/security_audit.sh
```

### View Logs

```bash
# Application logs
docker-compose logs -f

# System logs
journalctl -u ai-project-generator -f

# CI/CD logs
tail -f /var/log/cicd-*.log
```

### Health Checks

```bash
# Local health check
curl http://localhost:8020/health

# Production health check
curl http://$(cd aws/terraform && terraform output -raw load_balancer_dns)/health
```

## 📚 Documentation

- [README.md](README.md) - Main project documentation
- [CI_CD_README.md](CI_CD_README.md) - CI/CD documentation
- [aws/README.md](aws/README.md) - AWS deployment guide
- [aws/QUICK_START.md](aws/QUICK_START.md) - Quick start guide
- [.github/workflows/README.md](.github/workflows/README.md) - Workflows documentation

## 🎯 Quick Commands Reference

```bash
# Development
make install          # Install dependencies
make test             # Run tests
make lint             # Run linters
make format           # Format code
make security         # Security checks

# CI/CD
make ci               # Run CI pipeline
make cd               # Run CD pipeline
make build            # Build Docker image
make deploy           # Deploy to AWS

# AWS
make aws-deploy       # Deploy infrastructure
make aws-destroy      # Destroy infrastructure

# Utilities
make clean            # Clean temporary files
make pre-commit       # Run pre-commit hooks
```

## ✅ Verification Checklist

- [ ] Local development environment working
- [ ] Tests passing locally
- [ ] Pre-commit hooks installed
- [ ] AWS credentials configured
- [ ] GitHub secrets configured
- [ ] Infrastructure deployed
- [ ] Application deployed
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Backups configured
- [ ] CI/CD pipeline working

## 🆘 Troubleshooting

### Local Development Issues

```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Clear cache
make clean
```

### AWS Deployment Issues

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check Terraform state
cd aws/terraform
terraform state list

# Check Ansible inventory
cd aws/ansible
ansible-inventory -i inventory/ec2.ini --list
```

### CI/CD Issues

```bash
# Check workflow logs in GitHub Actions
# Run tests locally first
make test

# Validate scripts
./aws/scripts/script_validator.sh
```

## 📞 Support

For issues:
1. Check [Troubleshooting Guide](aws/docs/TROUBLESHOOTING.md)
2. Review workflow logs in GitHub Actions
3. Check application logs on EC2 instances
4. Review CloudWatch metrics and logs

