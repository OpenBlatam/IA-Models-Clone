# Circuit Breaker Implementation
# Using AWS X-Ray for distributed tracing and circuit breaker patterns

# X-Ray Daemon (for ECS)
resource "aws_ecs_task_definition" "xray_daemon" {
  family                   = "xray-daemon-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"

  container_definitions = jsonencode([{
    name  = "xray-daemon"
    image = "amazon/aws-xray-daemon:latest"
    portMappings = [{
      containerPort = 2000
      protocol      = "udp"
    }]
  }])

  tags = {
    Name = "xray-daemon-${var.environment}"
  }
}

# X-Ray Sampling Rule
resource "aws_xray_sampling_rule" "main" {
  rule_name      = "music-analyzer-sampling-${var.environment}"
  priority       = 10000
  version        = 1
  reservoir_size = 1
  fixed_rate     = 0.1

  service_name    = "music-analyzer-ai"
  service_type    = "*"
  host            = "*"
  http_method     = "*"
  url_path        = "*"
  resource_arn    = "*"
}

# CloudWatch Alarms for Circuit Breaker
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "music-analyzer-high-error-rate-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This metric monitors error rate"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  tags = {
    Name = "music-analyzer-high-error-rate-${var.environment}"
  }
}

resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "music-analyzer-high-latency-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Average"
  threshold           = 5.0
  alarm_description   = "This metric monitors response time"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  tags = {
    Name = "music-analyzer-high-latency-${var.environment}"
  }
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "music-analyzer-alerts-${var.environment}"

  tags = {
    Name = "music-analyzer-alerts-${var.environment}"
  }
}

# SNS Subscription (Email)
resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Auto Scaling based on Custom Metrics
resource "aws_appautoscaling_policy" "ecs_request_count" {
  name               = "music-analyzer-request-count-${var.environment}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    customized_metric_specification {
      metric_name = "RequestCount"
      namespace   = "AWS/ApplicationELB"
      statistic   = "Sum"

      dimensions = {
        LoadBalancer = aws_lb.main.arn_suffix
      }
    }
    target_value       = 1000.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# Health Check Enhancement
resource "aws_lb_target_group" "detailed_health" {
  name     = "music-analyzer-health-tg-${var.environment}"
  port     = 8010
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 10
    path                = "/health/detailed"
    protocol            = "HTTP"
    matcher             = "200"
  }

  deregistration_delay = 10

  tags = {
    Name = "music-analyzer-health-tg-${var.environment}"
  }
}




