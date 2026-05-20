resource "aws_elasticache_subnet_group" "redis" {
  count      = var.enable_elasticache ? 1 : 0
  name       = "${var.project_name}-redis-subnet-${var.environment}"
  subnet_ids = try(module.vpc.private_subnet_ids, aws_subnet.private[*].id)
}

resource "aws_elasticache_replication_group" "redis" {
  count = var.enable_elasticache ? 1 : 0

  replication_group_id       = "${var.project_name}-redis-${var.environment}"
  description                = "Redis cluster for ${var.project_name}"
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled           = true
  subnet_group_name          = aws_elasticache_subnet_group.redis[0].name
  security_group_ids         = [aws_security_group.redis[0].id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = false

  tags = {
    Name = "${var.project_name}-redis-${var.environment}"
  }
}

# Alternative: Single node Redis cluster (cheaper, less resilient)
resource "aws_elasticache_cluster" "redis" {
  count = var.enable_elasticache && var.environment != "production" ? 1 : 0

  cluster_id           = "${var.project_name}-redis-${var.environment}"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.redis[0].name
  security_group_ids   = [aws_security_group.redis[0].id]
  at_rest_encryption_enabled = true

  tags = {
    Name = "${var.project_name}-redis-${var.environment}"
  }
}

