# Advanced Terraform Features

This document describes the advanced features available in the Terraform configuration.

## 📋 Table of Contents

1. [Cost Optimization](#cost-optimization)
2. [Advanced IAM](#advanced-iam)
3. [Advanced CloudWatch](#advanced-cloudwatch)
4. [Disaster Recovery](#disaster-recovery)
5. [Workspaces](#workspaces)
6. [CI/CD Integration](#cicd-integration)
7. [Variable Validation](#variable-validation)
8. [Multi-Region Support](#multi-region-support)

## 💰 Cost Optimization

### Budget Alerts

Enable AWS budget alerts to monitor spending:

```hcl
enable_cost_budget      = true
monthly_budget_limit     = "500"
alert_email             = "admin@example.com"
```

### Spot Instances

Use spot instances for cost savings in non-production environments:

```hcl
enable_spot_instances    = true
spot_instance_type      = "t3.medium"
spot_max_price          = "0.05"
```

### Cost Allocation Tags

Automatic cost allocation tags are applied to all resources:

- `CostCenter`: Project or department
- `BillingProject`: Billing project name
- `Environment`: Environment name

## 🔐 Advanced IAM

### Secrets Manager Integration

Enable access to AWS Secrets Manager:

```hcl
enable_secrets_manager = true
secrets_manager_arns   = [
  "arn:aws:secretsmanager:us-east-1:123456789012:secret:app/db-password"
]
```

### Parameter Store Integration

Enable access to Systems Manager Parameter Store:

```hcl
enable_parameter_store = true
```

Parameters accessible at: `/ai-project-generator/production/*`

### ECR Access

Enable ECR access for container images:

```hcl
enable_ecr_access = true
```

## 📊 Advanced CloudWatch

### Dashboard

Automated CloudWatch dashboard with key metrics:

```hcl
enable_cloudwatch_dashboard = true
```

Includes:
- EC2 metrics (CPU, Network)
- ALB metrics (Response time, Request count)
- Auto Scaling metrics

### Log Insights

Pre-configured CloudWatch Log Insights queries:

```hcl
enable_cloudwatch_insights = true
```

Queries:
- Error logs
- Slow requests

### Anomaly Detection

CloudWatch anomaly detection for CPU:

```hcl
enable_anomaly_detection = true
```

### Log Metrics

Extract metrics from logs:

```hcl
enable_log_metrics = true
```

Metrics:
- ErrorCount
- WarningCount

### Advanced Alarms

Additional alarms for:
- High error rate (5xx errors)
- High response time

```hcl
enable_advanced_alarms = true
```

## 🛡️ Disaster Recovery

### Backup Bucket

S3 bucket for backups with lifecycle policies:

```hcl
enable_backup_bucket     = true
backup_retention_days    = 30
```

Features:
- Versioning enabled
- Encryption at rest
- Lifecycle transitions (Standard → IA → Glacier)

### Cross-Region Replication

Replicate backups to another region:

```hcl
enable_backup_replication    = true
backup_replication_region    = "us-west-2"
```

### EBS Snapshots

Automated EBS snapshots with DLM:

```hcl
enable_ebs_snapshots      = true
snapshot_retention_count   = 7
```

Features:
- Daily snapshots at 3:00 AM
- Automatic retention
- Tagged snapshots

## 🔄 Workspaces

Use Terraform workspaces for managing multiple environments:

```bash
# Create workspace
terraform workspace new dev
terraform workspace new staging
terraform workspace new production

# Switch workspace
terraform workspace select production

# List workspaces
terraform workspace list
```

Workspace-specific configurations:
- **dev**: t3.small, 1-2 instances, spot instances enabled
- **staging**: t3.medium, 1-3 instances
- **production**: t3.large, 2-5 instances

## 🚀 CI/CD Integration

### CodePipeline

Automated CI/CD pipeline:

```hcl
enable_codepipeline        = true
codestar_connection_arn    = "arn:aws:codestar-connections:..."
github_repository          = "owner/repo"
github_branch              = "main"
```

Pipeline stages:
1. **Source**: GitHub source
2. **Build**: CodeBuild
3. **Deploy**: CodeDeploy

### CodeBuild

Build configuration:
- Image: `aws/codebuild/standard:5.0`
- Compute: `BUILD_GENERAL1_SMALL`
- Privileged mode enabled

## ✅ Variable Validation

Enhanced input validation for all variables:

- **Region**: Valid AWS region
- **Environment**: dev, staging, or production
- **Instance Type**: Valid EC2 instance type
- **Ports**: Valid port range (1024-65535)
- **CIDR**: Valid CIDR block
- **Email**: Valid email format

See `variables_validation.tf` for details.

## 🌍 Multi-Region Support

### Provider Configuration

Primary and replica providers configured in `providers.tf`:

```hcl
provider "aws" {
  region = var.aws_region
}

provider "aws" {
  alias  = "replica"
  region = var.backup_replication_region
}
```

### Cross-Region Resources

Resources can be created in multiple regions:
- Backup replication
- Disaster recovery
- Multi-region deployments

## 📝 Usage Examples

### Complete Cost Optimization Setup

```hcl
# terraform.tfvars
enable_cost_budget      = true
monthly_budget_limit    = "500"
cost_center            = "Engineering"
billing_project        = "AI-Project-Generator"
enable_spot_instances  = true
spot_instance_type     = "t3.medium"
```

### Complete Monitoring Setup

```hcl
# terraform.tfvars
enable_cloudwatch_dashboard = true
enable_cloudwatch_insights  = true
enable_advanced_alarms      = true
enable_anomaly_detection    = true
enable_log_metrics          = true
enable_sns_alerts          = true
alert_email                = "admin@example.com"
```

### Complete Disaster Recovery Setup

```hcl
# terraform.tfvars
enable_backup_bucket         = true
backup_retention_days        = 90
enable_backup_replication    = true
backup_replication_region    = "us-west-2"
enable_ebs_snapshots         = true
snapshot_retention_count     = 30
```

## 🔗 Related Documentation

- [README_TERRAFORM.md](README_TERRAFORM.md) - Complete guide
- [IMPROVEMENTS_TERRAFORM.md](IMPROVEMENTS_TERRAFORM.md) - Improvements
- [SUMMARY_IMPROVEMENTS.md](SUMMARY_IMPROVEMENTS.md) - Summary

## ✅ Feature Checklist

- [x] Cost optimization (budgets, spot instances)
- [x] Advanced IAM (Secrets Manager, Parameter Store, ECR)
- [x] Advanced CloudWatch (dashboard, insights, anomaly detection)
- [x] Disaster recovery (backups, replication, snapshots)
- [x] Workspaces support
- [x] CI/CD integration
- [x] Variable validation
- [x] Multi-region support
- [x] Cost allocation tags
- [x] Advanced monitoring

---

**Version**: 2.1  
**Status**: Production Ready ✅

