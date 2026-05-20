# Advanced Infrastructure Configuration
# Includes: Auto-scaling, Load Balancing, Service Mesh, Message Brokers, High Availability

# VPC Configuration
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "music-analyzer-vpc-${var.environment}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "music-analyzer-igw-${var.environment}"
  }
}

# Public Subnets (Multi-AZ)
resource "aws_subnet" "public" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "music-analyzer-public-${count.index + 1}-${var.environment}"
    Type = "Public"
  }
}

# Private Subnets (Multi-AZ)
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + length(var.availability_zones))
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "music-analyzer-private-${count.index + 1}-${var.environment}"
    Type = "Private"
  }
}

# NAT Gateways (High Availability)
resource "aws_eip" "nat" {
  count  = length(var.availability_zones)
  domain = "vpc"

  tags = {
    Name = "music-analyzer-nat-eip-${count.index + 1}-${var.environment}"
  }
}

resource "aws_nat_gateway" "main" {
  count         = length(var.availability_zones)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "music-analyzer-nat-${count.index + 1}-${var.environment}"
  }

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "music-analyzer-public-rt-${var.environment}"
  }
}

resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = {
    Name = "music-analyzer-private-rt-${count.index + 1}-${var.environment}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "music-analyzer-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.environment == "production"
  enable_http2               = true
  enable_cross_zone_load_balancing = true

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.id
    enabled = true
  }

  tags = {
    Name = "music-analyzer-alb-${var.environment}"
  }
}

# ALB Security Group
resource "aws_security_group" "alb" {
  name        = "music-analyzer-alb-sg-${var.environment}"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "music-analyzer-alb-sg-${var.environment}"
  }
}

# ALB Target Group
resource "aws_lb_target_group" "main" {
  name     = "music-analyzer-tg-${var.environment}"
  port     = 8010
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
  }

  stickiness {
    enabled = true
    type    = "lb_cookie"
    cookie_duration = 86400
  }

  deregistration_delay = 30

  tags = {
    Name = "music-analyzer-tg-${var.environment}"
  }
}

# ALB Listener (HTTP - Redirect to HTTPS)
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# ALB Listener (HTTPS)
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.main.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.main.arn
  }
}

# ACM Certificate
resource "aws_acm_certificate" "main" {
  domain_name       = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = var.additional_domains

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "music-analyzer-cert-${var.environment}"
  }
}

# S3 Bucket for ALB Logs
resource "aws_s3_bucket" "alb_logs" {
  bucket = "music-analyzer-alb-logs-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "music-analyzer-alb-logs-${var.environment}"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  rule {
    id     = "delete-old-logs"
    status = "Enabled"

    expiration {
      days = 30
    }
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "music-analyzer-cluster-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "music-analyzer-cluster-${var.environment}"
  }
}

# ECS Service with Auto Scaling
resource "aws_ecs_service" "main" {
  name            = "music-analyzer-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = var.ecs_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.main.arn
    container_name   = "music-analyzer-ai"
    container_port   = 8010
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  enable_execute_command = true

  tags = {
    Name = "music-analyzer-service-${var.environment}"
  }
}

# ECS Auto Scaling
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = var.ecs_max_capacity
  min_capacity       = var.ecs_min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.main.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_cpu" {
  name               = "music-analyzer-cpu-autoscaling-${var.environment}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

resource "aws_appautoscaling_policy" "ecs_memory" {
  name               = "music-analyzer-memory-autoscaling-${var.environment}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value       = 80.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# ECS Security Group
resource "aws_security_group" "ecs" {
  name        = "music-analyzer-ecs-sg-${var.environment}"
  description = "Security group for ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "HTTP from ALB"
    from_port       = 8010
    to_port         = 8010
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "music-analyzer-ecs-sg-${var.environment}"
  }
}

# RDS PostgreSQL (Multi-AZ)
resource "aws_db_subnet_group" "main" {
  name       = "music-analyzer-db-subnet-${var.environment}"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "music-analyzer-db-subnet-${var.environment}"
  }
}

resource "aws_db_instance" "main" {
  identifier             = "music-analyzer-db-${var.environment}"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = var.rds_instance_class
  allocated_storage      = var.rds_allocated_storage
  storage_type           = "gp3"
  storage_encrypted      = true

  db_name  = "music_analyzer_db"
  username = var.rds_username
  password = var.rds_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  multi_az               = var.environment == "production"
  publicly_accessible    = false
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  performance_insights_enabled    = true
  deletion_protection            = var.environment == "production"

  tags = {
    Name = "music-analyzer-db-${var.environment}"
  }
}

# RDS Read Replicas (for production)
resource "aws_db_instance" "read_replica" {
  count              = var.environment == "production" ? var.rds_read_replica_count : 0
  identifier         = "music-analyzer-db-replica-${count.index + 1}-${var.environment}"
  replicate_source_db = aws_db_instance.main.identifier
  instance_class     = var.rds_instance_class

  publicly_accessible = false
  multi_az           = false

  tags = {
    Name = "music-analyzer-db-replica-${count.index + 1}-${var.environment}"
  }
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name        = "music-analyzer-rds-sg-${var.environment}"
  description = "Security group for RDS"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from ECS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "music-analyzer-rds-sg-${var.environment}"
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "redis" {
  name       = "music-analyzer-redis-subnet-${var.environment}"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "music-analyzer-redis-${var.environment}"
  description                = "Redis cluster for Music Analyzer AI"
  engine                     = "redis"
  engine_version             = "7.0"
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  num_cache_clusters         = var.redis_cluster_size

  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids          = [aws_security_group.redis.id]

  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  auth_token                  = var.redis_auth_token

  automatic_failover_enabled  = var.environment == "production"
  multi_az_enabled            = var.environment == "production"

  snapshot_retention_limit    = 5
  snapshot_window            = "03:00-05:00"

  tags = {
    Name = "music-analyzer-redis-${var.environment}"
  }
}

# Redis Security Group
resource "aws_security_group" "redis" {
  name        = "music-analyzer-redis-sg-${var.environment}"
  description = "Security group for ElastiCache Redis"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Redis from ECS"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "music-analyzer-redis-sg-${var.environment}"
  }
}

# SQS Queues for Async Processing
resource "aws_sqs_queue" "analysis_queue" {
  name                      = "music-analyzer-analysis-${var.environment}"
  visibility_timeout_seconds = 300
  message_retention_seconds  = 1209600  # 14 days
  receive_wait_time_seconds  = 20  # Long polling

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.analysis_dlq.arn
    maxReceiveCount     = 3
  })

  tags = {
    Name = "music-analyzer-analysis-queue-${var.environment}"
  }
}

resource "aws_sqs_queue" "analysis_dlq" {
  name = "music-analyzer-analysis-dlq-${var.environment}"

  message_retention_seconds = 1209600  # 14 days

  tags = {
    Name = "music-analyzer-analysis-dlq-${var.environment}"
  }
}

# EventBridge for Event-Driven Architecture
resource "aws_cloudwatch_event_rule" "analysis_complete" {
  name        = "music-analyzer-analysis-complete-${var.environment}"
  description = "Triggered when music analysis is complete"

  event_pattern = jsonencode({
    source      = ["music.analyzer"]
    detail-type = ["Analysis Complete"]
  })

  tags = {
    Name = "music-analyzer-analysis-complete-${var.environment}"
  }
}

# WAF for Security
resource "aws_wafv2_web_acl" "main" {
  name        = "music-analyzer-waf-${var.environment}"
  description = "WAF for Music Analyzer AI"
  scope       = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    override_action {
      none {}
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    override_action {
      none {}
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "KnownBadInputsMetric"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "RateLimitRule"
    priority = 3

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    action {
      block {}
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitMetric"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "MusicAnalyzerWAF"
    sampled_requests_enabled   = true
  }

  tags = {
    Name = "music-analyzer-waf-${var.environment}"
  }
}

resource "aws_wafv2_web_acl_association" "main" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Music Analyzer AI CDN"
  default_root_object = "index.html"

  aliases = var.domain_name != "" ? [var.domain_name] : []

  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "alb-origin"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "alb-origin"

    forwarded_values {
      query_string = true
      headers      = ["*"]

      cookies {
        forward = "all"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
  }

  # Cache behavior for API (no cache)
  ordered_cache_behavior {
    path_pattern     = "/music/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "alb-origin"

    forwarded_values {
      query_string = true
      headers      = ["*"]

      cookies {
        forward = "all"
      }
    }

    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
    compress               = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.main.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name = "music-analyzer-cloudfront-${var.environment}"
  }
}




