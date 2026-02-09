# Advanced Monitoring and Observability

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "music-analyzer-dashboard-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", { "stat": "Sum", "period": 60 }],
            [".", "HTTPCode_Target_2XX_Count", { "stat": "Sum", "period": 60 }],
            [".", "HTTPCode_Target_4XX_Count", { "stat": "Sum", "period": 60 }],
            [".", "HTTPCode_Target_5XX_Count", { "stat": "Sum", "period": 60 }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Request Metrics"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", { "stat": "Average", "period": 60 }],
            [".", "TargetResponseTime", { "stat": "p95", "period": 60 }],
            [".", "TargetResponseTime", { "stat": "p99", "period": 60 }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Response Time"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", { "stat": "Average", "period": 60 }],
            [".", "MemoryUtilization", { "stat": "Average", "period": 60 }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ECS Resource Utilization"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", { "stat": "Average", "period": 60 }],
            [".", "DatabaseConnections", { "stat": "Average", "period": 60 }],
            [".", "FreeableMemory", { "stat": "Average", "period": 60 }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "RDS Metrics"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ElastiCache", "CPUUtilization", { "stat": "Average", "period": 60 }],
            [".", "CacheHits", { "stat": "Sum", "period": 60 }],
            [".", "CacheMisses", { "stat": "Sum", "period": 60 }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Redis Metrics"
        }
      }
    ]
  })
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/music-analyzer-ai-${var.environment}"
  retention_in_days = var.log_retention_days

  kms_key_id = aws_kms_key.logs.arn

  tags = {
    Name = "music-analyzer-ecs-logs-${var.environment}"
  }
}

resource "aws_cloudwatch_log_group" "alb" {
  name              = "/aws/alb/music-analyzer-${var.environment}"
  retention_in_days = var.log_retention_days

  kms_key_id = aws_kms_key.logs.arn

  tags = {
    Name = "music-analyzer-alb-logs-${var.environment}"
  }
}

# KMS Key for Logs
resource "aws_kms_key" "logs" {
  description             = "KMS key for CloudWatch logs encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "music-analyzer-logs-key-${var.environment}"
  }
}

resource "aws_kms_alias" "logs" {
  name          = "alias/music-analyzer-logs-${var.environment}"
  target_key_id = aws_kms_key.logs.key_id
}

# OpenTelemetry Collector (for distributed tracing)
resource "aws_ecs_task_definition" "otel_collector" {
  family                   = "otel-collector-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"

  container_definitions = jsonencode([{
    name  = "otel-collector"
    image = "otel/opentelemetry-collector:latest"
    portMappings = [{
      containerPort = 4317
      protocol      = "tcp"
    }]
    environment = [
      {
        name  = "AWS_REGION"
        value = var.aws_region
      }
    ]
  }])

  tags = {
    Name = "otel-collector-${var.environment}"
  }
}

# Prometheus Metrics Export
resource "aws_prometheus_workspace" "main" {
  alias = "music-analyzer-prometheus-${var.environment}"

  tags = {
    Name = "music-analyzer-prometheus-${var.environment}"
  }
}

# Grafana Workspace
resource "aws_grafana_workspace" "main" {
  name                     = "music-analyzer-grafana-${var.environment}"
  account_access_type      = "CURRENT_ACCOUNT"
  authentication_providers = ["SAML"]
  permission_type          = "SERVICE_MANAGED"
  role_arn                 = aws_iam_role.grafana.arn

  data_sources = ["CLOUDWATCH", "PROMETHEUS", "XRAY"]

  tags = {
    Name = "music-analyzer-grafana-${var.environment}"
  }
}

# IAM Role for Grafana
resource "aws_iam_role" "grafana" {
  name = "music-analyzer-grafana-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "grafana.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "grafana" {
  role       = aws_iam_role.grafana.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess"
}




