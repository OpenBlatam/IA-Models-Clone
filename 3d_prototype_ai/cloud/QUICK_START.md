# 🚀 Quick Start Guide - AWS EC2 Deployment

Get your 3D Prototype AI application running on AWS EC2 in minutes!

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **SSH Key Pair** in AWS (or create a new one)
4. **Terraform** (optional, for IaC) or use the simple launch script

## Method 1: One-Click Deployment (Easiest)

### Step 1: Configure Environment

```bash
cd cloud
cp .env.example .env
# Edit .env with your AWS details
```

Update `.env`:
```env
AWS_REGION=us-east-1
AWS_INSTANCE_TYPE=t3.large
AWS_KEY_NAME=your-key-pair-name
AWS_KEY_PATH=~/.ssh/your-key-pair-name.pem
```

### Step 2: Run Deployment

```bash
chmod +x scripts/*.sh
./scripts/deploy.sh
```

That's it! The script will:
- Launch EC2 instance
- Configure the server
- Deploy your application
- Show you the access URL

## Method 2: Simple Launch Script

If you prefer a simpler approach:

```bash
cd cloud
cp .env.example .env
# Edit .env
./scripts/launch_ec2.sh
```

Then manually copy your application files and deploy.

## Method 3: Terraform (Infrastructure as Code)

### Step 1: Configure Terraform

```bash
cd cloud/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
```

### Step 2: Deploy

```bash
terraform init
terraform plan
terraform apply
```

### Step 3: Get Outputs

```bash
terraform output
```

## Method 4: CloudFormation

```bash
cd cloud/cloudformation
aws cloudformation deploy \
  --template-file stack.yaml \
  --stack-name 3d-prototype-ai \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    InstanceType=t3.large \
    KeyName=your-key-pair
```

## Method 5: Ansible (Configuration Management)

### Step 1: Launch Instance

Use Method 1 or 2 to launch an instance first.

### Step 2: Configure Inventory

```bash
cd cloud/ansible/inventory
cp ec2.ini.example ec2.ini
# Edit ec2.ini with your instance IP
```

### Step 3: Run Playbook

```bash
cd cloud/ansible
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml
```

## After Deployment

### Access Your Application

Once deployed, you'll get:
- **Application URL**: `http://<instance-ip>:8030`
- **Health Check**: `http://<instance-ip>/health`
- **API Docs**: `http://<instance-ip>/docs`

### Verify Deployment

```bash
./scripts/health_check.sh --ip <instance-ip>
```

### SSH Access

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
```

## Next Steps

1. **Configure Domain** (optional)
   - Point your domain to the instance IP
   - Set up SSL certificate with Let's Encrypt

2. **Set Up Monitoring**
   - Configure CloudWatch alarms
   - Set up log aggregation

3. **Backup Strategy**
   - Configure automated backups
   - Set up S3 for storage

4. **Scaling** (if needed)
   - Set up Auto Scaling Group
   - Configure Load Balancer

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues and solutions.

## Cost Estimation

Approximate monthly costs (us-east-1):
- **t3.micro**: ~$7/month (free tier eligible)
- **t3.small**: ~$15/month
- **t3.medium**: ~$30/month
- **t3.large**: ~$60/month (recommended for production)

*Prices may vary by region and usage*

## Security Best Practices

1. **Restrict SSH Access**
   - Update `ssh_allowed_cidr_blocks` in terraform.tfvars
   - Use your specific IP address

2. **Use IAM Roles**
   - Don't store AWS credentials on instance
   - Use IAM roles for EC2

3. **Enable Encryption**
   - EBS volumes are encrypted by default
   - Use HTTPS for application access

4. **Regular Updates**
   - Keep system packages updated
   - Update application regularly

## Support

For issues or questions:
- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- Review main project [README.md](../README.md)
- Create an issue on GitHub

