# Local Values - Computed values used throughout configuration

locals {
  # Common tags
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Application = "AI Project Generator"
    CreatedAt   = timestamp()
  }

  # Resource naming
  name_prefix = "${var.project_name}-${var.environment}"

  # Availability zones
  availability_zones = data.aws_availability_zones.available.names

  # Account and region info
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name

  # S3 bucket names
  terraform_state_bucket = "${var.project_name}-terraform-state-${var.environment}"
  alb_logs_bucket        = "${var.project_name}-alb-logs-${var.environment}"

  # Redis configuration
  redis_endpoint = var.enable_elasticache ? (
    try(aws_elasticache_replication_group.redis[0].configuration_endpoint_address,
        aws_elasticache_cluster.redis[0].cache_nodes[0].address, "")
  ) : "localhost"

  redis_url = "redis://${local.redis_endpoint}:6379"

  # User data template variables
  user_data_vars = {
    project_name = var.project_name
    environment  = var.environment
    app_port     = var.app_port
    redis_url    = local.redis_url
  }
}

