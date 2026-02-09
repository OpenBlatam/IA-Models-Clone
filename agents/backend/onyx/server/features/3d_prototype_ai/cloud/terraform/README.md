# Terraform Infrastructure as Code

This directory contains Terraform configuration for deploying the 3D Prototype AI application on AWS EC2, following DevOps best practices.

## 📁 Structure

```
terraform/
├── main.tf              # Main configuration and provider setup
├── versions.tf          # Version constraints
├── variables.tf         # Input variables
├── outputs.tf           # Output values
├── locals.tf            # Local computed values
├── data.tf              # Data sources
├── networking.tf        # VPC, subnets, routing
├── security.tf          # Security groups, IAM
├── compute.tf           # EC2 instance, alarms
├── storage.tf           # S3 buckets for backups
├── terraform.tfvars.example  # Example variables
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan Deployment

```bash
terraform plan
```

### 4. Apply Configuration

```bash
terraform apply
```

### 5. View Outputs

```bash
terraform output
```

## 📋 Configuration

### Required Variables

- `key_name`: AWS key pair name for SSH access

### Optional Variables

See `variables.tf` for complete list. Common ones:

- `aws_region`: AWS region (default: us-east-1)
- `instance_type`: EC2 instance type (default: t3.large)
- `environment`: Environment name (default: prod)
- `create_vpc`: Create new VPC (default: false)
- `allocate_elastic_ip`: Allocate Elastic IP (default: false)

## 🏗️ Resources Created

### Networking (if create_vpc = true)
- VPC
- Internet Gateway
- Public Subnet
- Route Table

### Security
- Security Group (HTTP, HTTPS, SSH, Application port)
- IAM Role for EC2
- IAM Policies (CloudWatch, S3)

### Compute
- EC2 Instance
- Elastic IP (optional)
- CloudWatch Alarms (CPU, Memory)

### Storage (if enable_s3_backup = true)
- S3 Bucket for backups
- S3 Bucket versioning
- S3 Bucket encryption
- S3 Bucket lifecycle policy

## 🔒 Security Features

- **Encrypted EBS volumes**: All volumes encrypted by default
- **IMDSv2**: Instance metadata service v2 required
- **Least privilege IAM**: Minimal permissions for EC2 role
- **Security groups**: Restrictive ingress rules
- **S3 encryption**: Backups encrypted at rest

## 📊 Monitoring

### CloudWatch Alarms

- **CPU Utilization**: Alerts when CPU > threshold (default: 80%)
- **Memory Utilization**: Alerts when memory > threshold (default: 85%, requires CloudWatch agent)

### Enable Alarms

```hcl
enable_cloudwatch_alarms = true
enable_memory_alarm      = true
sns_topic_arn           = "arn:aws:sns:region:account:topic-name"
```

## 💾 Backups

### S3 Backups

Enable S3 bucket for backups:

```hcl
enable_s3_backup        = true
backup_retention_days   = 30
```

Features:
- Versioning enabled
- Encryption at rest
- Lifecycle policies
- Public access blocked

## 🔧 Advanced Configuration

### Custom AMI

```hcl
ami_id = "ami-xxxxxxxxxxxxx"
```

### Additional EBS Volumes

```hcl
additional_volumes = [
  {
    device_name = "/dev/sdf"
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
    iops        = 3000
    throughput  = 125
  }
]
```

### Custom Volume Performance

```hcl
volume_iops       = 3000
volume_throughput = 125
```

## 📤 Outputs

After deployment, get outputs:

```bash
# All outputs
terraform output

# Specific output
terraform output instance_public_ip
terraform output application_url
terraform output ssh_command
```

## 🔄 State Management

### Remote State (Recommended)

Configure S3 backend in `backend.tfvars`:

```hcl
bucket         = "your-terraform-state-bucket"
key            = "3d-prototype-ai/terraform.tfstate"
region         = "us-east-1"
encrypt        = true
dynamodb_table = "terraform-state-lock"
```

Initialize with backend:

```bash
terraform init -backend-config=backend.tfvars
```

### Local State

Default behavior, state stored in `terraform.tfstate`.

## 🧹 Cleanup

### Destroy Resources

```bash
terraform destroy
```

### Selective Destruction

Use `-target` flag:

```bash
terraform destroy -target=aws_instance.app
```

## ✅ Best Practices

1. **Version Control**: Commit `.tf` files, ignore `.tfstate`
2. **Remote State**: Use S3 backend for team collaboration
3. **State Locking**: Use DynamoDB for state locking
4. **Variables**: Use `.tfvars` files, never commit secrets
5. **Tags**: All resources tagged for cost tracking
6. **Modularity**: Separate concerns into different files
7. **Validation**: Run `terraform validate` before apply
8. **Formatting**: Run `terraform fmt` before commit

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
terraform plan -out=tfplan
terraform show tfplan
```

## 🐛 Troubleshooting

### Common Issues

1. **State Lock**: If state is locked, check DynamoDB table
2. **Provider Version**: Ensure AWS provider version matches
3. **Permissions**: Verify IAM permissions for Terraform
4. **Key Pair**: Ensure key pair exists in AWS
5. **VPC/Subnet**: Verify VPC and subnet IDs if using existing

### Debug Mode

```bash
TF_LOG=DEBUG terraform apply
```

## 📚 Additional Resources

- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

---

**Last Updated**: 2024-01-XX
**Terraform Version**: >= 1.5.0
**AWS Provider Version**: ~> 5.0

