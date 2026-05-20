# Detailed Deployment Guide - AI Project Generator

Complete step-by-step guide for deploying the AI Project Generator to AWS EC2.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Infrastructure Deployment](#infrastructure-deployment)
4. [Application Deployment](#application-deployment)
5. [Verification](#verification)
6. [Updates and Maintenance](#updates-and-maintenance)

## Prerequisites

### Required Software

1. **AWS CLI** (v2.x)
   ```bash
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   aws configure
   ```

2. **Terraform** (>= 1.0)
   ```bash
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   ```

3. **Ansible** (>= 2.9)
   ```bash
   pip3 install ansible boto3
   ansible-galaxy collection install amazon.aws community.general
   ```

4. **SSH Key Pair**
   ```bash
   aws ec2 create-key-pair \
     --key-name ai-project-generator-key \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/ai-project-generator-key.pem
   chmod 400 ~/.ssh/ai-project-generator-key.pem
   ```

## Initial Setup

### 1. Configure Terraform Variables

```bash
cd aws/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region = "us-east-1"
project_name = "ai-project-generator"
environment = "production"
instance_type = "t3.medium"  # Use t3.large+ for ML workloads
key_name = "ai-project-generator-key"
app_port = 8020
enable_elasticache = false  # Set to true for production
```

### 2. Configure Ansible Variables

Edit `ansible/group_vars/all.yml`:

```yaml
project_name: "ai-project-generator"
environment: "production"
app_port: 8020
redis_url: "redis://localhost:6379"
mcp_secret_key: "your-secret-key-here"
```

## Infrastructure Deployment

### Option 1: Automated Deployment

```bash
cd aws/scripts
export DEPLOY_INFRASTRUCTURE=true
export DEPLOY_APPLICATION=false
export ENVIRONMENT=production
./deploy.sh
```

### Option 2: Manual Terraform Deployment

```bash
cd aws/terraform
terraform init
terraform plan
terraform apply
```

### What Gets Created

- VPC with public/private subnets (2 AZs)
- Internet Gateway and NAT Gateways
- Security Groups (ALB, EC2, Redis)
- Application Load Balancer
- Auto Scaling Group
- Optional ElastiCache Redis
- IAM roles for EC2 instances

## Application Deployment

### Option 1: Automated Deployment

```bash
cd aws/scripts
export DEPLOY_INFRASTRUCTURE=false
export DEPLOY_APPLICATION=true
export ENVIRONMENT=production
./deploy.sh
```

### Option 2: Manual Ansible Deployment

```bash
cd aws/ansible
ansible-playbook \
  -i inventory/ec2.ini \
  playbooks/deploy.yml \
  -e "environment=production" \
  --ask-become-pass
```

### Deployment Process

1. **Common Setup**: System updates, firewall, timezone
2. **Docker**: Install Docker and Docker Compose
3. **Python**: Install Python 3.11 and virtual environment
4. **Redis**: Install and configure Redis server
5. **Nginx**: Configure reverse proxy
6. **Application**: Deploy application code and start services

## Verification

### 1. Check Application Health

```bash
# Get load balancer DNS
cd aws/terraform
terraform output load_balancer_dns

# Test health endpoint
curl http://$(terraform output -raw load_balancer_dns)/health
```

### 2. Check Application Logs

```bash
# SSH into instance
ssh -i ~/.ssh/ai-project-generator-key.pem ubuntu@<instance-ip>

# Docker logs
docker-compose -f /opt/ai-project-generator/docker-compose.yml logs -f

# System logs
sudo journalctl -u ai-project-generator -f
```

### 3. Verify Services

```bash
# Check Nginx
sudo systemctl status nginx

# Check Redis
redis-cli ping

# Check Docker
docker ps

# Check application
curl http://localhost:8020/health
```

## Updates and Maintenance

### Update Application

```bash
cd aws/ansible
ansible-playbook \
  -i inventory/ec2.ini \
  playbooks/update.yml \
  -e "git_branch=main"
```

### Scale Instances

```bash
cd aws/terraform
terraform apply -var="desired_capacity=4"
```

### View Logs

```bash
# Application logs
ansible all -i inventory/ec2.ini -m shell \
  -a "docker-compose -f /opt/ai-project-generator/docker-compose.yml logs --tail=100"

# System logs
ansible all -i inventory/ec2.ini -m shell \
  -a "sudo journalctl -u nginx -n 50"
```

## Application Endpoints

- **Health**: `http://<load-balancer-dns>/health`
- **Metrics**: `http://<load-balancer-dns>/metrics`
- **API Status**: `http://<load-balancer-dns>/api/v1/status`
- **API Generate**: `http://<load-balancer-dns>/api/v1/generate`
- **API Queue**: `http://<load-balancer-dns>/api/v1/queue`

## Cleanup

To destroy all resources:

```bash
cd aws/terraform
terraform destroy
```

**Warning**: This will delete all infrastructure. Backup important data first.

