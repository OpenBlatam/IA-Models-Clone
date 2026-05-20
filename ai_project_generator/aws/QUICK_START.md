# Quick Start Guide - AI Project Generator AWS Deployment

Get your AI Project Generator running on AWS EC2 in minutes.

## Prerequisites Checklist

- [ ] AWS account with appropriate permissions
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Terraform installed (>= 1.0)
- [ ] Ansible installed (>= 2.9)
- [ ] SSH key pair created in AWS

## 5-Minute Setup

### 1. Create SSH Key Pair (2 minutes)

```bash
# Create key pair in AWS
aws ec2 create-key-pair \
  --key-name ai-project-generator-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/ai-project-generator-key.pem

# Set permissions
chmod 400 ~/.ssh/ai-project-generator-key.pem
```

### 2. Configure Variables (1 minute)

```bash
cd aws/terraform
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
# Minimum required: key_name = "ai-project-generator-key"
```

### 3. Deploy Infrastructure (2 minutes)

```bash
cd aws/terraform
terraform init
terraform plan
terraform apply
```

### 4. Deploy Application

```bash
cd ../ansible

# Install Ansible collections (first time only)
ansible-galaxy collection install amazon.aws community.general

# Deploy application
ansible-playbook \
  -i inventory/ec2.ini \
  playbooks/deploy.yml \
  --ask-become-pass
```

### 5. Access Your Application

```bash
# Get load balancer DNS
cd ../terraform
terraform output load_balancer_dns

# Open in browser
# http://<load-balancer-dns>/health
```

## One-Command Deployment

For automated deployment:

```bash
cd aws/scripts
export ENVIRONMENT=production
export DEPLOY_INFRASTRUCTURE=true
export DEPLOY_APPLICATION=true
./deploy.sh
```

## Common Commands

### Check Application Status

```bash
# Health check
curl http://$(cd terraform && terraform output -raw load_balancer_dns)/health

# Or use the script
cd scripts
./health_check.sh all
```

### View Logs

```bash
# SSH into instance
ssh -i ~/.ssh/ai-project-generator-key.pem ubuntu@<instance-ip>

# Application logs (Docker)
docker-compose -f /opt/ai-project-generator/docker-compose.yml logs -f

# System logs
sudo journalctl -u ai-project-generator -f
```

### Update Application

```bash
cd aws/ansible
ansible-playbook \
  -i inventory/ec2.ini \
  playbooks/update.yml \
  --ask-become-pass
```

### Scale Instances

```bash
cd aws/terraform
terraform apply -var="desired_capacity=4"
```

## Application Endpoints

- **Health Check**: `http://<load-balancer-dns>/health`
- **Metrics**: `http://<load-balancer-dns>/metrics`
- **API**: `http://<load-balancer-dns>/api/v1/`
- **Status**: `http://<load-balancer-dns>/api/v1/status`

## Troubleshooting

### Can't connect to instances?

1. Check security group allows SSH from your IP
2. Verify key pair name matches
3. Check instance is running: `aws ec2 describe-instances`

### Application not starting?

1. Check logs: `docker-compose logs` or `journalctl -u ai-project-generator`
2. Verify environment variables: `cat /opt/ai-project-generator/.env`
3. Check Redis is running: `redis-cli ping`
4. Check disk space: `df -h`

### Load balancer returning 502?

1. Check target group health in AWS console
2. Verify security group allows traffic from ALB
3. Test health endpoint on instance: `curl http://localhost:8020/health`

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Review [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for step-by-step instructions
- Configure monitoring and alerts in CloudWatch
- Set up automated backups
- Configure SSL/TLS certificates for HTTPS

