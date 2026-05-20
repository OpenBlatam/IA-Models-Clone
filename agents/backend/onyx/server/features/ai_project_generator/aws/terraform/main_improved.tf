# Main Terraform Configuration - Improved with Modules
# This file uses modules for better organization and reusability

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(
      {
        Project     = var.project_name
        Environment = var.environment
        ManagedBy   = "Terraform"
        Application = "AI Project Generator"
      },
      var.tags
    )
  }
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  project_name            = var.project_name
  environment             = var.environment
  vpc_cidr                = var.vpc_cidr
  availability_zones_count = 2
  tags                    = var.tags
}

# Application Load Balancer Module
module "alb" {
  source = "./modules/alb"

  project_name              = var.project_name
  environment               = var.environment
  vpc_id                    = module.vpc.vpc_id
  subnet_ids                = module.vpc.public_subnet_ids
  security_group_ids        = [aws_security_group.alb.id]
  target_port               = var.app_port
  health_check_path         = var.health_check_path
  enable_https              = var.enable_https
  certificate_arn            = var.certificate_arn
  enable_deletion_protection = var.environment == "production"
  enable_access_logs        = var.enable_alb_access_logs
  access_logs_bucket        = var.alb_access_logs_bucket
  tags                      = var.tags
}

# EC2 Module
module "ec2" {
  source = "./modules/ec2"

  project_name         = var.project_name
  environment          = var.environment
  instance_type        = var.instance_type
  key_name             = var.key_name
  security_group_ids   = [aws_security_group.app.id]
  subnet_ids           = module.vpc.private_subnet_ids
  target_group_arns    = [module.alb.target_group_arn]
  min_size             = var.min_size
  max_size             = var.max_size
  desired_capacity     = var.desired_capacity
  root_volume_size     = var.root_volume_size
  root_volume_type     = var.root_volume_type
  enable_monitoring    = var.enable_monitoring
  user_data            = templatefile("${path.module}/../scripts/ec2_user_data.sh", {
    project_name = var.project_name
    environment  = var.environment
    app_port     = var.app_port
    redis_url    = var.enable_elasticache ? "redis://${try(aws_elasticache_replication_group.redis[0].configuration_endpoint_address, aws_elasticache_cluster.redis[0].cache_nodes[0].address)}:6379" : "redis://localhost:6379"
  })
  s3_bucket_arns       = var.s3_bucket_arns
  tags                 = var.tags
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/ec2/${var.project_name}-${var.environment}"
  retention_in_days = var.log_retention_days

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-logs-${var.environment}"
    }
  )
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${var.project_name}-high-cpu-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = var.enable_sns_alerts ? [aws_sns_topic.alerts[0].arn] : []

  dimensions = {
    AutoScalingGroupName = module.ec2.autoscaling_group_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "low_cpu" {
  alarm_name          = "${var.project_name}-low-cpu-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 20
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = var.enable_sns_alerts ? [aws_sns_topic.alerts[0].arn] : []

  dimensions = {
    AutoScalingGroupName = module.ec2.autoscaling_group_name
  }

  tags = var.tags
}

# SNS Topic for Alerts (Optional)
resource "aws_sns_topic" "alerts" {
  count = var.enable_sns_alerts ? 1 : 0

  name = "${var.project_name}-alerts-${var.environment}"

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-alerts-${var.environment}"
    }
  )
}

resource "aws_sns_topic_subscription" "email" {
  count = var.enable_sns_alerts && var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Note: Security groups, Redis, and networking resources are defined in separate files
# They reference either module outputs (if using modules) or direct resources (legacy)

