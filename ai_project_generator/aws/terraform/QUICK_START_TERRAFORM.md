# Terraform Quick Start Guide

Quick reference for using the improved Terraform configuration.

## 🚀 Initial Setup

### 1. Create Backend Resources (First Time Only)

```bash
cd aws/terraform

# Create S3 bucket and DynamoDB table for state
terraform apply -target=aws_s3_bucket.terraform_state \
                -target=aws_dynamodb_table.terraform_state_lock
```

### 2. Configure Backend

Create `backend.hcl`:
```hcl
bucket         = "ai-project-generator-terraform-state-production"
key            = "terraform.tfstate"
region         = "us-east-1"
dynamodb_table = "ai-project-generator-terraform-state-lock-production"
encrypt        = true
```

### 3. Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 4. Initialize

```bash
terraform init -backend-config=backend.hcl
```

## 📦 Using Modules (Recommended)

### Switch to Modular Configuration

```bash
# Backup current main.tf
cp main.tf main_legacy.tf

# Use improved modular configuration
cp main_improved.tf main.tf

# Reinitialize
terraform init -upgrade
```

## 🔧 Common Commands

### Plan and Apply

```bash
# Review changes
terraform plan

# Apply with auto-approve (use with caution)
terraform apply -auto-approve

# Apply with plan file
terraform plan -out=tfplan
terraform apply tfplan
```

### State Management

```bash
# List resources
terraform state list

# Show resource
terraform state show aws_instance.app

# Import resource
terraform import aws_instance.app i-1234567890abcdef0
```

### Outputs

```bash
# Show all outputs
terraform output

# Specific output
terraform output load_balancer_dns
```

## 🎯 Module Usage Examples

### VPC Module

```hcl
module "vpc" {
  source = "./modules/vpc"
  
  project_name = "ai-project-generator"
  environment  = "production"
  vpc_cidr     = "10.0.0.0/16"
}
```

### ALB Module

```hcl
module "alb" {
  source = "./modules/alb"
  
  project_name    = "ai-project-generator"
  environment     = "production"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.public_subnet_ids
  target_port     = 8020
  enable_https    = true
  certificate_arn = "arn:aws:acm:..."
}
```

### EC2 Module

```hcl
module "ec2" {
  source = "./modules/ec2"
  
  project_name      = "ai-project-generator"
  environment       = "production"
  instance_type     = "t3.medium"
  subnet_ids        = module.vpc.private_subnet_ids
  target_group_arns = [module.alb.target_group_arn]
}
```

## 🔐 Security Best Practices

1. **State Encryption**: Always enable encryption
2. **State Locking**: Use DynamoDB for locking
3. **IAM Roles**: Use IAM roles, not hardcoded credentials
4. **Security Groups**: Follow least privilege
5. **Encrypted Volumes**: Enable EBS encryption

## 📊 Monitoring

### CloudWatch

```bash
# View logs
aws logs tail /aws/ec2/ai-project-generator-production --follow

# View metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=AutoScalingGroupName,Value=ai-project-generator-asg-production
```

### SNS Alerts

```hcl
# Enable in terraform.tfvars
enable_sns_alerts = true
alert_email       = "admin@example.com"
```

## 🆘 Troubleshooting

### State Lock Issues

```bash
# Force unlock (use with caution)
terraform force-unlock <lock-id>
```

### Backend Migration

```bash
# Migrate from local to S3
terraform init -migrate-state
```

### Module Updates

```bash
# Update modules
terraform get -update

# Verify module sources
terraform init -upgrade
```

## 📚 Next Steps

1. Review [README_TERRAFORM.md](README_TERRAFORM.md) for complete guide
2. Check [IMPROVEMENTS_TERRAFORM.md](IMPROVEMENTS_TERRAFORM.md) for improvements
3. See [CHANGELOG_TERRAFORM.md](CHANGELOG_TERRAFORM.md) for version history

