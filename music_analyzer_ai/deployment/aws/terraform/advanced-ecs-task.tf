# Advanced ECS Task Definition with all optimizations

resource "aws_ecs_task_definition" "main" {
  family                   = "music-analyzer-task-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "music-analyzer-ai"
      image = "${aws_ecr_repository.main.repository_url}:latest"

      portMappings = [{
        containerPort = 8010
        protocol      = "tcp"
      }]

      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "SPOTIFY_CLIENT_ID"
          value = var.spotify_client_id
        },
        {
          name  = "SPOTIFY_CLIENT_SECRET"
          value = var.spotify_client_secret
        },
        {
          name  = "DATABASE_URL"
          value = "postgresql://${var.rds_username}:${var.rds_password}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
        },
        {
          name  = "REDIS_URL"
          value = "redis://${aws_elasticache_replication_group.redis.configuration_endpoint_address}:6379"
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "AWS_XRAY_DAEMON_ADDRESS"
          value = "xray-daemon:2000"
        }
      ]

      secrets = [
        {
          name      = "SPOTIFY_CLIENT_SECRET"
          valueFrom = aws_secretsmanager_secret.spotify_client_secret.arn
        },
        {
          name      = "RDS_PASSWORD"
          valueFrom = aws_secretsmanager_secret.rds_password.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8010/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      ulimits = [{
        name      = "nofile"
        softLimit = 65536
        hardLimit = 65536
      }]

      dockerLabels = {
        "com.music-analyzer.version" = "2.21.0"
        "com.music-analyzer.environment" = var.environment
      }
    },
    {
      name  = "xray-daemon"
      image = "amazon/aws-xray-daemon:latest"

      portMappings = [{
        containerPort = 2000
        protocol      = "udp"
      }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "xray"
        }
      }
    }
  ])

  tags = {
    Name = "music-analyzer-task-${var.environment}"
  }
}

# ECR Repository
resource "aws_ecr_repository" "main" {
  name                 = "music-analyzer-ai-${var.environment}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name = "music-analyzer-ecr-${var.environment}"
  }
}

# ECR Lifecycle Policy
resource "aws_ecr_lifecycle_policy" "main" {
  repository = aws_ecr_repository.main.name

  policy = jsonencode({
    rules = [
      {
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
      }
    ]
  })
}

# Secrets Manager
resource "aws_secretsmanager_secret" "spotify_client_secret" {
  name        = "music-analyzer/spotify-client-secret-${var.environment}"
  description = "Spotify API Client Secret"

  kms_key_id = aws_kms_key.secrets.arn

  tags = {
    Name = "music-analyzer-spotify-secret-${var.environment}"
  }
}

resource "aws_secretsmanager_secret_version" "spotify_client_secret" {
  secret_id     = aws_secretsmanager_secret.spotify_client_secret.id
  secret_string = var.spotify_client_secret
}

resource "aws_secretsmanager_secret" "rds_password" {
  name        = "music-analyzer/rds-password-${var.environment}"
  description = "RDS Master Password"

  kms_key_id = aws_kms_key.secrets.arn

  tags = {
    Name = "music-analyzer-rds-secret-${var.environment}"
  }
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  secret_id     = aws_secretsmanager_secret.rds_password.id
  secret_string = var.rds_password
}

# KMS Key for Secrets
resource "aws_kms_key" "secrets" {
  description             = "KMS key for Secrets Manager"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "music-analyzer-secrets-key-${var.environment}"
  }
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/music-analyzer-secrets-${var.environment}"
  target_key_id = aws_kms_key.secrets.key_id
}

# IAM Roles
resource "aws_iam_role" "ecs_execution" {
  name = "music-analyzer-ecs-execution-role-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "music-analyzer-ecs-execution-secrets-${var.environment}"
  role = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "kms:Decrypt"
        ]
        Resource = [
          aws_secretsmanager_secret.spotify_client_secret.arn,
          aws_secretsmanager_secret.rds_password.arn,
          aws_kms_key.secrets.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "music-analyzer-ecs-task-role-${var.environment}"

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
  name = "music-analyzer-ecs-task-policy-${var.environment}"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sns:Publish"
        ]
        Resource = "*"
      }
    ]
  })
}

# Additional Variables
variable "ecs_task_cpu" {
  description = "ECS task CPU units (256, 512, 1024, etc.)"
  type        = number
  default     = 1024
}

variable "ecs_task_memory" {
  description = "ECS task memory in MB"
  type        = number
  default     = 2048
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}




