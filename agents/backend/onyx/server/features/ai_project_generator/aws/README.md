# AWS EC2 Deployment for AI Project Generator

Complete infrastructure and deployment configuration for deploying the AI Project Generator to AWS EC2 instances.

## 📋 Overview

This deployment package provides:
- **Infrastructure as Code** with Terraform
- **Configuration Management** with Ansible
- **Automated Deployment** scripts
- **Production-ready** setup with monitoring and scaling

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Application Load Balancer                   │
│                    (Port 80/443)                         │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌────────▼────────┐
│  EC2 Instance  │          │  EC2 Instance    │
│  (Auto Scaling)│          │  (Auto Scaling)  │
│                │          │                  │
│  FastAPI:8020  │          │  FastAPI:8020    │
│  Nginx:80      │          │  Nginx:80        │
│  Redis:6379    │          │  Redis:6379      │
└───────┬────────┘          └────────┬─────────┘
        │                            │
        └──────────────┬─────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   ElastiCache Redis (Optional)│
        └──────────────────────────────┘
```

## 📁 Directory Structure

```
aws/
├── terraform/              # Infrastructure as Code (Improved v2.0)
│   ├── main.tf            # Provider configuration (legacy)
│   ├── main_improved.tf   # Modular configuration (recommended)
│   ├── variables.tf       # Variable definitions
│   ├── outputs.tf        # Output values
│   ├── backend.tf        # Backend configuration
│   ├── s3_backend.tf     # S3 backend resources
│   ├── data.tf           # Data sources
│   ├── locals.tf         # Local values
│   ├── versions.tf       # Version constraints
│   ├── modules/          # Reusable modules
│   │   ├── vpc/         # VPC module
│   │   ├── alb/         # ALB module
│   │   └── ec2/         # EC2 module
│   ├── ec2.tf            # EC2 and Auto Scaling (legacy)
│   ├── networking.tf     # VPC and networking (legacy)
│   ├── security_groups.tf # Security groups
│   ├── redis.tf          # ElastiCache Redis (optional)
│   ├── README_TERRAFORM.md # Complete Terraform guide
│   └── QUICK_START_TERRAFORM.md # Quick start guide
│   └── terraform.tfvars.example
├── ansible/               # Configuration Management
│   ├── playbooks/
│   │   ├── deploy.yml     # Main deployment
│   │   ├── setup.yml      # Initial setup
│   │   └── update.yml     # Update application
│   ├── roles/
│   │   ├── common/        # Common configuration
│   │   ├── docker/        # Docker setup
│   │   ├── python/        # Python 3.11 setup
│   │   ├── redis/        # Redis setup
│   │   ├── nginx/         # Nginx reverse proxy
│   │   └── app/           # Application deployment
│   ├── inventory/ec2.ini  # Dynamic EC2 inventory
│   ├── group_vars/all.yml # Common variables
│   └── ansible.cfg        # Ansible configuration
├── scripts/               # Deployment scripts
│   ├── deploy.sh         # Main deployment script
│   ├── ec2_user_data.sh  # EC2 initialization
│   ├── health_check.sh   # Health check script
│   └── setup_environment.sh
├── docs/                  # Documentation
│   ├── DEPLOYMENT.md     # Deployment guide
│   └── TROUBLESHOOTING.md
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

1. **AWS CLI** configured
   ```bash
   aws configure
   ```

2. **Terraform** (>= 1.0)
   ```bash
   terraform version
   ```

3. **Ansible** (>= 2.9)
   ```bash
   ansible --version
   pip install ansible boto3
   ```

4. **SSH Key Pair** in AWS
   ```bash
   aws ec2 create-key-pair \
     --key-name ai-project-generator-key \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/ai-project-generator-key.pem
   chmod 400 ~/.ssh/ai-project-generator-key.pem
   ```

### Step 1: Configure Variables

```bash
cd aws/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### Step 2: Deploy Infrastructure

**Option A: Using Improved Modular Configuration (Recommended)**

```bash
cd aws/terraform

# Create backend resources (first time only)
terraform apply -target=aws_s3_bucket.terraform_state \
                -target=aws_dynamodb_table.terraform_state_lock

# Configure backend
cp backend.hcl.example backend.hcl
# Edit backend.hcl with your values

# Switch to modular configuration
cp main.tf main_legacy.tf
cp main_improved.tf main.tf

# Initialize and deploy
terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

**Option B: Using Legacy Configuration**

```bash
cd aws/terraform
terraform init
terraform plan
terraform apply
```

> **Note**: See [terraform/README_TERRAFORM.md](terraform/README_TERRAFORM.md) for complete Terraform guide and improvements.

### Step 3: Deploy Application

```bash
cd ../ansible
ansible-galaxy collection install amazon.aws community.general
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml
```

### Step 4: Access Application

```bash
# Get load balancer DNS
cd ../terraform
terraform output load_balancer_dns

# Access application
curl http://$(terraform output -raw load_balancer_dns)/health
```

## 📝 Configuration

### Environment Variables

Key environment variables for the application:

- `REDIS_URL`: Redis connection string
- `MCP_SECRET_KEY`: Secret key for MCP server
- `OTLP_ENDPOINT`: OpenTelemetry endpoint (optional)
- `ENVIRONMENT`: Environment name (production/staging/dev)

### Instance Types

Recommended instance types:
- **Development**: `t3.medium` (2 vCPU, 4 GB RAM)
- **Production**: `t3.large` or `t3.xlarge` (for ML workloads)
- **High Performance**: `g4dn.xlarge` (for GPU workloads)

## 🔧 Features

### Infrastructure
- ✅ **Modular Terraform Configuration** (v2.0) - Reusable modules for VPC, ALB, EC2
- ✅ **S3 Backend** - Remote state storage with versioning
- ✅ **DynamoDB State Locking** - Prevent concurrent modifications
- ✅ **VPC with public/private subnets** - Multi-AZ deployment
- ✅ **Application Load Balancer** - High availability
- ✅ **Auto Scaling Group** - Automatic scaling based on metrics
- ✅ **Security Groups** - Least privilege access
- ✅ **Optional ElastiCache Redis** - Managed Redis cluster
- ✅ **CloudWatch Monitoring** - Logs, metrics, and alarms
- ✅ **SNS Alerts** - Optional email notifications

### Application
- ✅ FastAPI application on port 8020
- ✅ Nginx reverse proxy
- ✅ Docker containerization
- ✅ Health checks
- ✅ Logging and monitoring
- ✅ **Automatic deployment** on push to main ⭐
- ✅ **Enhanced CI/CD** with testing, security scanning, rollback ⭐⭐
- ✅ **Blue-green deployments** for zero-downtime ⭐
- ✅ **Canary deployments** for gradual rollouts ⭐
- ✅ **Feature flags** for A/B testing ⭐
- ✅ **Performance testing** before deployment ⭐
- ✅ **Database migrations** with rollback ⭐
- ✅ **Multi-region deployment** for global availability ⭐⭐
- ✅ **Automated disaster recovery** with failover ⭐⭐
- ✅ **Advanced logging & tracing** with CloudWatch ⭐
- ✅ **Cost optimization** and monitoring ⭐
- ✅ **Prometheus & Grafana** for advanced observability ⭐⭐
- ✅ **Security hardening** with fail2ban, auditd, AppArmor ⭐⭐
- ✅ **API Gateway** with rate limiting and versioning ⭐
- ✅ **Chaos engineering** for resilience testing ⭐

### Security
- ✅ Encrypted EBS volumes
- ✅ Security groups with restricted access
- ✅ Secrets management ready
- ✅ SSL/TLS ready (with ACM certificate)

## 📊 Monitoring

### CloudWatch Metrics
- CPU utilization
- Memory usage
- Network I/O
- Application health

### Application Metrics
- Request rate
- Response time
- Error rate
- Queue length

## 🔄 Updates

### Automatic Deployment (Recommended)

**GitHub Actions** - Automatic deployment on push to main branch:

1. Configure GitHub Secrets (see [CI/CD Setup Guide](docs/CI_CD_SETUP.md))
2. Push to main branch
3. Deployment runs automatically via GitHub Actions

**Webhook/Polling** - EC2 instance automatically deploys:

```bash
# Setup on EC2 instance
cd aws/scripts
sudo bash setup_webhook_listener.sh
sudo systemctl start github-webhook-deploy
```

### Manual Update

To manually update the application:

```bash
cd aws/ansible
ansible-playbook -i inventory/ec2.ini playbooks/update.yml
```

## 🧹 Cleanup

To destroy all resources:

```bash
cd aws/terraform
terraform destroy
```

## 📚 Documentation

### General
- [Deployment Guide](docs/DEPLOYMENT.md) - Detailed deployment steps
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Quick Start](QUICK_START.md) - Quick deployment guide
- [CI/CD Setup](docs/CI_CD_SETUP.md) - **Automatic deployment setup** ⭐
- [Enhanced CI/CD](docs/ENHANCED_CI_CD.md) - **Advanced CI/CD features** ⭐⭐
- [Advanced Deployment Strategies](docs/ADVANCED_DEPLOYMENT_STRATEGIES.md) - **Canary, Blue-Green, Feature Flags** ⭐⭐
- [Enterprise Features](docs/ENTERPRISE_FEATURES.md) - **Multi-Region, DR, Logging, Cost Optimization** ⭐⭐⭐
- [Observability & Security](docs/OBSERVABILITY_SECURITY.md) - **Prometheus, Grafana, Security Hardening, API Gateway, Chaos Engineering** ⭐⭐⭐

### Terraform (v2.0 - Improved)
- [Terraform Guide](terraform/README_TERRAFORM.md) - Complete Terraform documentation
- [Terraform Quick Start](terraform/QUICK_START_TERRAFORM.md) - Quick reference
- [Terraform Improvements](terraform/IMPROVEMENTS_TERRAFORM.md) - What's new in v2.0
- [Terraform Changelog](terraform/CHANGELOG_TERRAFORM.md) - Version history

## 🆘 Support

For issues or questions:
1. Check [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Review CloudWatch logs
3. Check application logs on instances

