variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "ai-project-generator"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "instance_type" {
  description = "EC2 instance type. Recommended: t3.medium for dev, t3.large+ for production with ML workloads"
  type        = string
  default     = "t3.medium"
}

variable "key_name" {
  description = "AWS key pair name for EC2 instances"
  type        = string
}

variable "min_size" {
  description = "Minimum number of instances in auto-scaling group"
  type        = number
  default     = 1
}

variable "max_size" {
  description = "Maximum number of instances in auto-scaling group"
  type        = number
  default     = 3
}

variable "desired_capacity" {
  description = "Desired number of instances in auto-scaling group"
  type        = number
  default     = 2
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_public_ip" {
  description = "Enable public IP for EC2 instances"
  type        = bool
  default     = true
}

variable "root_volume_size" {
  description = "Root volume size in GB. Increase for ML models storage"
  type        = number
  default     = 50
}

variable "root_volume_type" {
  description = "Root volume type"
  type        = string
  default     = "gp3"
}

variable "enable_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "enable_elasticache" {
  description = "Enable ElastiCache Redis (recommended for production)"
  type        = bool
  default     = false
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "app_port" {
  description = "Application port (FastAPI default: 8020)"
  type        = number
  default     = 8020
}

variable "health_check_path" {
  description = "Health check endpoint path"
  type        = string
  default     = "/health"
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

variable "enable_kms_encryption" {
  description = "Enable KMS encryption for S3 backend"
  type        = bool
  default     = false
}

variable "create_s3_backend" {
  description = "Create S3 bucket and DynamoDB table for Terraform state"
  type        = bool
  default     = true
}

variable "s3_bucket_arns" {
  description = "S3 bucket ARNs for EC2 IAM policy"
  type        = list(string)
  default     = []
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_https" {
  description = "Enable HTTPS listener on ALB"
  type        = bool
  default     = false
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
  default     = ""
}

variable "enable_alb_access_logs" {
  description = "Enable ALB access logs"
  type        = bool
  default     = false
}

variable "alb_access_logs_bucket" {
  description = "S3 bucket for ALB access logs"
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "enable_sns_alerts" {
  description = "Enable SNS alerts"
  type        = bool
  default     = false
}

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
  default     = ""
}

variable "use_default_vpc" {
  description = "Use default VPC instead of creating new one"
  type        = bool
  default     = false
}

# Cost Optimization Variables
variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = ""
}

variable "billing_project" {
  description = "Billing project name"
  type        = string
  default     = ""
}

variable "enable_cost_budget" {
  description = "Enable AWS budget alerts"
  type        = bool
  default     = false
}

variable "monthly_budget_limit" {
  description = "Monthly budget limit in USD"
  type        = string
  default     = "100"
}

variable "enable_spot_instances" {
  description = "Enable spot instances for cost savings"
  type        = bool
  default     = false
}

variable "spot_instance_type" {
  description = "Spot instance type (if different from instance_type)"
  type        = string
  default     = ""
}

variable "spot_max_price" {
  description = "Maximum price for spot instances"
  type        = string
  default     = ""
}

# Advanced IAM Variables
variable "enable_secrets_manager" {
  description = "Enable Secrets Manager access"
  type        = bool
  default     = false
}

variable "secrets_manager_arns" {
  description = "Secrets Manager ARNs"
  type        = list(string)
  default     = []
}

variable "enable_parameter_store" {
  description = "Enable Systems Manager Parameter Store access"
  type        = bool
  default     = false
}

variable "enable_ecr_access" {
  description = "Enable ECR access for container images"
  type        = bool
  default     = false
}

variable "create_autoscaling_service_role" {
  description = "Create service-linked role for Auto Scaling"
  type        = bool
  default     = false
}

# Advanced CloudWatch Variables
variable "enable_cloudwatch_dashboard" {
  description = "Enable CloudWatch dashboard"
  type        = bool
  default     = true
}

variable "enable_cloudwatch_insights" {
  description = "Enable CloudWatch Log Insights queries"
  type        = bool
  default     = false
}

variable "enable_advanced_alarms" {
  description = "Enable advanced CloudWatch alarms"
  type        = bool
  default     = false
}

variable "enable_anomaly_detection" {
  description = "Enable CloudWatch anomaly detection"
  type        = bool
  default     = false
}

variable "enable_log_metrics" {
  description = "Enable CloudWatch log metric filters"
  type        = bool
  default     = false
}

# Disaster Recovery Variables
variable "enable_backup_bucket" {
  description = "Enable S3 bucket for backups"
  type        = bool
  default     = false
}

variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 30
}

variable "enable_backup_replication" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = false
}

variable "backup_replication_region" {
  description = "Region for backup replication"
  type        = string
  default     = "us-west-2"
}

variable "enable_ebs_snapshots" {
  description = "Enable automated EBS snapshots"
  type        = bool
  default     = false
}

variable "snapshot_retention_count" {
  description = "Number of snapshots to retain"
  type        = number
  default     = 7
}

# CI/CD Variables
variable "enable_codepipeline" {
  description = "Enable AWS CodePipeline for CI/CD"
  type        = bool
  default     = false
}

variable "codestar_connection_arn" {
  description = "CodeStar connection ARN for GitHub"
  type        = string
  default     = ""
}

variable "github_repository" {
  description = "GitHub repository (owner/repo)"
  type        = string
  default     = ""
}

variable "github_branch" {
  description = "GitHub branch to deploy"
  type        = string
  default     = "main"
}

variable "ecs_cluster_name" {
  description = "ECS cluster name (if using ECS)"
  type        = string
  default     = ""
}

variable "ecs_service_name" {
  description = "ECS service name (if using ECS)"
  type        = string
  default     = ""
}

