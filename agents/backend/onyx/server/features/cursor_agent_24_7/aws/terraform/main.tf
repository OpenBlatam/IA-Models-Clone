# Terraform configuration para Cursor Agent 24/7 en AWS
# =====================================================

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Backend S3 (opcional, descomentar para usar)
  # backend "s3" {
  #   bucket = "cursor-agent-terraform-state"
  #   key    = "terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "cursor-agent-24-7"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# VPC y Networking (opcional, usar VPC existente si está disponible)
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ECR Repository
resource "aws_ecr_repository" "cursor_agent" {
  name                 = "cursor-agent-24-7"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "AES256"
  }
  
  lifecycle_policy {
    policy = jsonencode({
      rules = [{
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }]
    })
  }
}

# DynamoDB Table para estado
resource "aws_dynamodb_table" "agent_state" {
  name           = "cursor-agent-state"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "id"
  
  attribute {
    name = "id"
    type = "S"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  point_in_time_recovery {
    enabled = true
  }
  
  server_side_encryption {
    enabled = true
  }
  
  tags = {
    Name = "cursor-agent-state"
  }
}

# DynamoDB Table para caché
resource "aws_dynamodb_table" "agent_cache" {
  name           = "cursor-agent-cache"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "key"
  
  attribute {
    name = "key"
    type = "S"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  point_in_time_recovery {
    enabled = true
  }
  
  server_side_encryption {
    enabled = true
  }
  
  tags = {
    Name = "cursor-agent-cache"
  }
}

# ElastiCache Redis (opcional, para caché mejorado)
resource "aws_elasticache_subnet_group" "redis" {
  name       = "cursor-agent-redis-subnet"
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "cursor-agent-redis"
  description                = "Redis cache for cursor-agent-24-7"
  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  num_cache_clusters         = 1
  automatic_failover_enabled = false
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "cursor-agent-redis"
  }
}

# Security Group para Redis
resource "aws_security_group" "redis" {
  name        = "cursor-agent-redis-sg"
  description = "Security group for Redis cache"
  vpc_id      = data.aws_vpc.default.id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "cursor-agent-redis-sg"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "cursor_agent" {
  name              = "/aws/cursor-agent-24-7"
  retention_in_days = 30
  
  tags = {
    Name = "cursor-agent-logs"
  }
}

# IAM Role para ECS Task
resource "aws_iam_role" "ecs_task" {
  name = "cursor-agent-ecs-task-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task" {
  name = "cursor-agent-ecs-task-policy"
  role = aws_iam_role.ecs_task.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.agent_state.arn,
          aws_dynamodb_table.agent_cache.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.cursor_agent.arn}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# ECS Cluster
resource "aws_ecs_cluster" "cursor_agent" {
  name = "cursor-agent-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Name = "cursor-agent-cluster"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "cursor_agent" {
  family                   = "cursor-agent-24-7"
  network_mode             = "awsvpc"
  requires_compatibilities  = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn        = aws_iam_role.ecs_task.arn
  task_role_arn             = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([{
    name  = "cursor-agent"
    image = "${aws_ecr_repository.cursor_agent.repository_url}:latest"
    
    portMappings = [{
      containerPort = 8024
      protocol      = "tcp"
    }]
    
    environment = [
      {
        name  = "AWS_REGION"
        value = var.aws_region
      },
      {
        name  = "DYNAMODB_TABLE_NAME"
        value = aws_dynamodb_table.agent_state.name
      },
      {
        name  = "CACHE_TYPE"
        value = "elasticache"
      },
      {
        name  = "REDIS_ENDPOINT"
        value = aws_elasticache_replication_group.redis.configuration_endpoint_address
      }
    ]
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.cursor_agent.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8024/api/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
  
  tags = {
    Name = "cursor-agent-task"
  }
}

# Security Group para ECS
resource "aws_security_group" "ecs" {
  name        = "cursor-agent-ecs-sg"
  description = "Security group for ECS service"
  vpc_id      = data.aws_vpc.default.id
  
  ingress {
    from_port   = 8024
    to_port     = 8024
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Ajustar según necesidad
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "cursor-agent-ecs-sg"
  }
}

# Application Load Balancer
resource "aws_lb" "cursor_agent" {
  name               = "cursor-agent-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ecs.id]
  subnets            = data.aws_subnets.default.ids
  
  enable_deletion_protection = false
  
  tags = {
    Name = "cursor-agent-alb"
  }
}

resource "aws_lb_target_group" "cursor_agent" {
  name        = "cursor-agent-tg"
  port        = 8024
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/api/health"
    protocol            = "HTTP"
    matcher             = "200"
  }
  
  tags = {
    Name = "cursor-agent-tg"
  }
}

resource "aws_lb_listener" "cursor_agent" {
  load_balancer_arn = aws_lb.cursor_agent.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.cursor_agent.arn
  }
}

# ECS Service
resource "aws_ecs_service" "cursor_agent" {
  name            = "cursor-agent-service"
  cluster         = aws_ecs_cluster.cursor_agent.id
  task_definition = aws_ecs_task_definition.cursor_agent.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.cursor_agent.arn
    container_name   = "cursor-agent"
    container_port   = 8024
  }
  
  depends_on = [aws_lb_listener.cursor_agent]
  
  tags = {
    Name = "cursor-agent-service"
  }
}

# Outputs
output "ecr_repository_url" {
  value = aws_ecr_repository.cursor_agent.repository_url
}

output "alb_dns_name" {
  value = aws_lb.cursor_agent.dns_name
}

output "dynamodb_state_table" {
  value = aws_dynamodb_table.agent_state.name
}

output "dynamodb_cache_table" {
  value = aws_dynamodb_table.agent_cache.name
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.redis.configuration_endpoint_address
}




