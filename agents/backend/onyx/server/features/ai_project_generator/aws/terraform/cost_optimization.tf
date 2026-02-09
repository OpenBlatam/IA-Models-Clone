# Cost Optimization Resources

# Cost allocation tags
locals {
  cost_tags = {
    CostCenter     = var.cost_center != "" ? var.cost_center : var.project_name
    BillingProject = var.billing_project != "" ? var.billing_project : var.project_name
    Environment    = var.environment
    ManagedBy      = "Terraform"
  }
}

# Merge cost tags with existing tags
locals {
  all_tags = merge(
    var.tags,
    local.cost_tags,
    {
      Project     = var.project_name
      Application = "AI Project Generator"
    }
  )
}

# Budget alerts (optional)
resource "aws_budgets_budget" "cost" {
  count = var.enable_cost_budget ? 1 : 0

  name              = "${var.project_name}-budget-${var.environment}"
  budget_type       = "COST"
  limit_amount      = var.monthly_budget_limit
  limit_unit        = "USD"
  time_period_start = "2024-01-01_00:00"
  time_unit         = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email != "" ? [var.alert_email] : []
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = var.alert_email != "" ? [var.alert_email] : []
  }

  tags = local.all_tags
}

# Spot instance configuration for cost savings (optional)
resource "aws_launch_template" "spot" {
  count = var.enable_spot_instances ? 1 : 0

  name_prefix   = "${var.project_name}-spot-${var.environment}-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.spot_instance_type != "" ? var.spot_instance_type : var.instance_type
  key_name      = var.key_name

  vpc_security_group_ids = [aws_security_group.app.id]

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price          = var.spot_max_price != "" ? var.spot_max_price : null
      spot_instance_type = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size           = var.root_volume_size
      volume_type           = var.root_volume_type
      delete_on_termination = true
      encrypted             = true
    }
  }

  monitoring {
    enabled = var.enable_monitoring
  }

  user_data = base64encode(templatefile("${path.module}/../scripts/ec2_user_data.sh", {
    project_name = var.project_name
    environment  = var.environment
    app_port     = var.app_port
    redis_url    = var.enable_elasticache ? "redis://${try(aws_elasticache_replication_group.redis[0].configuration_endpoint_address, aws_elasticache_cluster.redis[0].cache_nodes[0].address)}:6379" : "redis://localhost:6379"
  }))

  iam_instance_profile {
    name = aws_iam_instance_profile.app.name
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(
      local.all_tags,
      {
        Name        = "${var.project_name}-spot-${var.environment}"
        InstanceType = "spot"
      }
    )
  }
}

# Reserved instance recommendations (informational)
# Note: Reserved instances should be purchased manually through AWS Console
# This is just a placeholder for documentation

