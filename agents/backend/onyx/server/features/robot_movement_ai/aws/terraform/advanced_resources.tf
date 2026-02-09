# Advanced AWS Resources
# =====================
# Additional resources for advanced features: Kafka, Prometheus, Grafana, etc.

# MSK (Managed Streaming for Apache Kafka)
resource "aws_msk_cluster" "main" {
  count = var.enable_kafka ? 1 : 0
  
  cluster_name           = "${var.project_name}-kafka"
  kafka_version          = "3.5.1"
  number_of_broker_nodes = 3

  broker_node_group_info {
    instance_type   = "kafka.m5.large"
    client_subnets  = aws_subnet.private[*].id
    security_groups = [aws_security_group.kafka[0].id]
    storage_info {
      ebs_storage_info {
        volume_size = 100
      }
    }
  }

  encryption_info {
    encryption_at_rest_kms_key_id = aws_kms_key.kafka[0].arn
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  client_authentication {
    sasl {
      iam = true
    }
    tls {
      certificate_authority_arns = []
    }
  }

  tags = {
    Name = "${var.project_name}-kafka"
  }
}

# KMS Key for Kafka encryption
resource "aws_kms_key" "kafka" {
  count = var.enable_kafka ? 1 : 0
  
  description             = "KMS key for Kafka encryption"
  deletion_window_in_days = 10

  tags = {
    Name = "${var.project_name}-kafka-key"
  }
}

resource "aws_kms_alias" "kafka" {
  count = var.enable_kafka ? 1 : 0
  
  name          = "alias/${var.project_name}-kafka"
  target_key_id = aws_kms_key.kafka[0].key_id
}

# Security Group for Kafka
resource "aws_security_group" "kafka" {
  count = var.enable_kafka ? 1 : 0
  
  name        = "${var.project_name}-kafka-sg"
  description = "Security group for MSK cluster"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 9092
    to_port         = 9098
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
    Name = "${var.project_name}-kafka-sg"
  }
}

# ECS Service for Celery Workers
resource "aws_ecs_task_definition" "celery_worker" {
  count = var.enable_celery ? 1 : 0
  
  family                   = "${var.project_name}-celery-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "celery-worker"
      image = "${aws_ecr_repository.main.repository_url}:latest"
      command = [
        "celery",
        "-A", "aws.workers.celery_config",
        "worker",
        "--loglevel=info",
        "--concurrency=4"
      ]
      environment = [
        {
          name  = "REDIS_URL"
          value = var.enable_redis ? aws_elasticache_cluster.main[0].cache_nodes[0].address : "redis://localhost:6379"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "celery"
        }
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-celery-worker"
  }
}

resource "aws_ecs_service" "celery_worker" {
  count = var.enable_celery ? 1 : 0
  
  name            = "${var.project_name}-celery-worker-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.celery_worker[0].arn
  desired_count   = 2
  launch_type      = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  tags = {
    Name = "${var.project_name}-celery-worker-service"
  }
}

# ECS Service for Celery Beat (Scheduler)
resource "aws_ecs_task_definition" "celery_beat" {
  count = var.enable_celery ? 1 : 0
  
  family                   = "${var.project_name}-celery-beat"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "celery-beat"
      image = "${aws_ecr_repository.main.repository_url}:latest"
      command = [
        "celery",
        "-A", "aws.workers.celery_config",
        "beat",
        "--loglevel=info"
      ]
      environment = [
        {
          name  = "REDIS_URL"
          value = var.enable_redis ? aws_elasticache_cluster.main[0].cache_nodes[0].address : "redis://localhost:6379"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "celery-beat"
        }
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-celery-beat"
  }
}

resource "aws_ecs_service" "celery_beat" {
  count = var.enable_celery ? 1 : 0
  
  name            = "${var.project_name}-celery-beat-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.celery_beat[0].arn
  desired_count   = 1
  launch_type      = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  tags = {
    Name = "${var.project_name}-celery-beat-service"
  }
}

# API Gateway with Rate Limiting
resource "aws_apigatewayv2_api" "main" {
  count = var.enable_api_gateway ? 1 : 0
  
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"
  description   = "API Gateway for Robot Movement AI"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["*"]
    allow_headers = ["*"]
    max_age       = 300
  }

  tags = {
    Name = "${var.project_name}-api"
  }
}

# API Gateway Stage with Rate Limiting
resource "aws_apigatewayv2_stage" "main" {
  count = var.enable_api_gateway ? 1 : 0
  
  api_id      = aws_apigatewayv2_api.main[0].id
  name        = "prod"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway[0].arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway" {
  count = var.enable_api_gateway ? 1 : 0
  
  name              = "/aws/apigateway/${var.project_name}"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-api-gateway-logs"
  }
}

# WAF (Web Application Firewall) for DDoS Protection
resource "aws_wafv2_web_acl" "main" {
  count = var.enable_waf ? 1 : 0
  
  name        = "${var.project_name}-waf"
  description = "WAF for Robot Movement AI"
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
      metric_name                = "KnownBadInputsRuleSetMetric"
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
      metric_name                = "RateLimitRuleMetric"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-waf-metric"
    sampled_requests_enabled   = true
  }

  tags = {
    Name = "${var.project_name}-waf"
  }
}

# Associate WAF with ALB
resource "aws_wafv2_web_acl_association" "alb" {
  count = var.enable_waf ? 1 : 0
  
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.main[0].arn
}















