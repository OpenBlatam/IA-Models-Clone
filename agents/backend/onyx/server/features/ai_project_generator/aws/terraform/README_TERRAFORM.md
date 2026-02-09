# Terraform Configuration Guide

Complete guide for the improved Terraform infrastructure configuration.

## 📋 Structure

```
terraform/
├── main.tf                    # Main configuration (backward compatible)
├── main_improved.tf          # Improved modular configuration
├── variables.tf              # Variable definitions
├── outputs.tf                # Output values
├── backend.tf                # Backend configuration
├── s3_backend.tf             # S3 backend resources
├── data.tf                   # Data sources
├── locals.tf                 # Local values
├── versions.tf               # Version constraints
├── ec2.tf                    # EC2 configuration (legacy)
├── networking.tf              # Networking (legacy)
├── security_groups.tf        # Security groups
├── redis.tf                  # ElastiCache Redis
├── modules/                  # Reusable modules
│   ├── ec2/                 # EC2 module
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── alb/                 # ALB module
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── vpc/                 # VPC module
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
└── terraform.tfvars.example # Example variables
```

## 🚀 Quick Start

### 1. Initialize Backend (First Time)

```bash
cd aws/terraform

# Create S3 backend resources (one-time setup)
terraform apply -target=aws_s3_bucket.terraform_state \
                -target=aws_dynamodb_table.terraform_state_lock

# Configure backend
terraform init -backend-config=backend.hcl
```

### 2. Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 3. Plan and Apply

```bash
# Review changes
terraform plan

# Apply infrastructure
terraform apply
```

## 🔧 Using Modules (Recommended)

The improved configuration uses modules for better organization:

### Switch to Modular Configuration

```bash
# Backup current main.tf
mv main.tf main_legacy.tf

# Use improved modular configuration
mv main_improved.tf main.tf

# Reinitialize
terraform init -upgrade
```

### Module Benefits

- **Reusability**: Modules can be reused across projects
- **Maintainability**: Easier to maintain and update
- **Testability**: Modules can be tested independently
- **Organization**: Clear separation of concerns

## 📦 Available Modules

### 1. VPC Module (`modules/vpc/`)

Complete VPC setup with:
- VPC with DNS support
- Public and private subnets
- Internet Gateway
- NAT Gateways (one per AZ)
- Route tables

**Usage:**
```hcl
module "vpc" {
  source = "./modules/vpc"
  
  project_name = "ai-project-generator"
  environment  = "production"
  vpc_cidr     = "10.0.0.0/16"
}
```

### 2. ALB Module (`modules/alb/`)

Application Load Balancer with:
- HTTP and HTTPS listeners
- Target groups
- Health checks
- Access logs
- Session stickiness

**Usage:**
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

### 3. EC2 Module (`modules/ec2/`)

EC2 Auto Scaling Group with:
- Launch template
- Auto Scaling policies
- CloudWatch alarms
- IAM roles
- User data support

**Usage:**
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

## 🔐 Backend Configuration

### S3 Backend with State Locking

**Create backend resources:**
```bash
terraform apply -target=aws_s3_bucket.terraform_state \
                -target=aws_dynamodb_table.terraform_state_lock
```

**Configure backend (`backend.hcl`):**
```hcl
bucket         = "ai-project-generator-terraform-state-production"
key            = "terraform.tfstate"
region         = "us-east-1"
dynamodb_table = "ai-project-generator-terraform-state-lock-production"
encrypt        = true
```

**Initialize:**
```bash
terraform init -backend-config=backend.hcl
```

## 📊 Outputs

### View All Outputs

```bash
terraform output
```

### Specific Outputs

```bash
# Load balancer DNS
terraform output load_balancer_dns

# VPC ID
terraform output vpc_id

# Redis endpoint
terraform output redis_endpoint
```

## 🔄 State Management

### Import Existing Resources

```bash
terraform import aws_instance.app i-1234567890abcdef0
```

### State Operations

```bash
# List resources
terraform state list

# Show resource
terraform state show aws_instance.app

# Move resource
terraform state mv aws_instance.app module.ec2.aws_instance.app
```

### Remote State

```bash
# Pull remote state
terraform state pull > terraform.tfstate

# Push state
terraform state push terraform.tfstate
```

## 🛠️ Best Practices

### 1. Use Modules

- Organize code into reusable modules
- Keep modules focused and single-purpose
- Document module inputs and outputs

### 2. State Management

- Use S3 backend for team collaboration
- Enable versioning on state bucket
- Use DynamoDB for state locking
- Encrypt state at rest

### 3. Variables

- Use variables for all configurable values
- Provide defaults where appropriate
- Use variable validation
- Document variables

### 4. Outputs

- Output important resource identifiers
- Use descriptions for all outputs
- Group related outputs

### 5. Security

- Use IAM roles for EC2 instances
- Enable encryption at rest
- Use security groups with least privilege
- Enable VPC flow logs

## 📝 Example: Complete Setup

```bash
# 1. Create backend resources
terraform apply -target=aws_s3_bucket.terraform_state \
                -target=aws_dynamodb_table.terraform_state_lock

# 2. Configure backend
cat > backend.hcl <<EOF
bucket         = "ai-project-generator-terraform-state-production"
key            = "terraform.tfstate"
region         = "us-east-1"
dynamodb_table = "ai-project-generator-terraform-state-lock-production"
encrypt        = true
EOF

# 3. Initialize
terraform init -backend-config=backend.hcl

# 4. Plan
terraform plan -out=tfplan

# 5. Apply
terraform apply tfplan
```

## 🔍 Validation

### Validate Configuration

```bash
terraform validate
```

### Format Code

```bash
terraform fmt -recursive
```

### Check Plan

```bash
terraform plan -detailed-exitcode
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

### Module Issues

```bash
# Update modules
terraform get -update

# Verify module sources
terraform init -upgrade
```

## 📚 Additional Resources

- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

