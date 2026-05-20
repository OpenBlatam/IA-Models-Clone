# Advanced CloudWatch Configuration

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  count          = var.enable_cloudwatch_dashboard ? 1 : 0
  dashboard_name = "${var.project_name}-${var.environment}"

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
            ["AWS/EC2", "CPUUtilization", "AutoScalingGroupName", try(module.ec2.autoscaling_group_name, aws_autoscaling_group.app.name)],
            [".", "NetworkIn", ".", "."],
            [".", "NetworkOut", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "EC2 Metrics"
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
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", try(module.alb.load_balancer_arn_suffix, aws_lb.main.arn_suffix)],
            [".", "RequestCount", ".", "."],
            [".", "HTTPCode_Target_2XX_Count", ".", "."],
            [".", "HTTPCode_Target_4XX_Count", ".", "."],
            [".", "HTTPCode_Target_5XX_Count", ".", "."]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "ALB Metrics"
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
            ["AWS/AutoScaling", "GroupDesiredCapacity", "AutoScalingGroupName", try(module.ec2.autoscaling_group_name, aws_autoscaling_group.app.name)],
            [".", "GroupInServiceInstances", ".", "."],
            [".", "GroupTotalInstances", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Auto Scaling Metrics"
        }
      }
    ]
  })
}

# CloudWatch Log Insights Queries
resource "aws_cloudwatch_query_definition" "error_logs" {
  count = var.enable_cloudwatch_insights ? 1 : 0

  name = "${var.project_name}-error-logs-${var.environment}"

  log_group_names = [
    aws_cloudwatch_log_group.app.name
  ]

  query_string = <<-QUERY
    fields @timestamp, @message
    | filter @message like /ERROR/
    | sort @timestamp desc
    | limit 100
  QUERY
}

resource "aws_cloudwatch_query_definition" "slow_requests" {
  count = var.enable_cloudwatch_insights ? 1 : 0

  name = "${var.project_name}-slow-requests-${var.environment}"

  log_group_names = [
    aws_cloudwatch_log_group.app.name
  ]

  query_string = <<-QUERY
    fields @timestamp, @message
    | filter @message like /duration/
    | parse @message /duration=(?<duration>\d+)/
    | filter duration > 1000
    | sort duration desc
    | limit 100
  QUERY
}

# CloudWatch Composite Alarms
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  count = var.enable_advanced_alarms ? 1 : 0

  alarm_name          = "${var.project_name}-high-error-rate-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This metric monitors 5xx errors"
  alarm_actions       = var.enable_sns_alerts ? [aws_sns_topic.alerts[0].arn] : []

  dimensions = {
    LoadBalancer = try(module.alb.load_balancer_arn_suffix, aws_lb.main.arn_suffix)
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "high_response_time" {
  count = var.enable_advanced_alarms ? 1 : 0

  alarm_name          = "${var.project_name}-high-response-time-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Average"
  threshold           = 2.0
  alarm_description   = "This metric monitors response time"
  alarm_actions       = var.enable_sns_alerts ? [aws_sns_topic.alerts[0].arn] : []

  dimensions = {
    LoadBalancer = try(module.alb.load_balancer_arn_suffix, aws_lb.main.arn_suffix)
  }

  tags = var.tags
}

# CloudWatch Anomaly Detection
resource "aws_cloudwatch_metric_alarm" "cpu_anomaly" {
  count = var.enable_anomaly_detection ? 1 : 0

  alarm_name          = "${var.project_name}-cpu-anomaly-${var.environment}"
  comparison_operator = "GreaterThanUpperThreshold"
  evaluation_periods  = 2
  threshold_metric_id = "e1"

  metric_query {
    id          = "e1"
    expression  = "ANOMALY_DETECTION_BAND(m1, 2)"
    label       = "CPUUtilization (Expected)"
    return_data = "true"
  }

  metric_query {
    id          = "m1"
    return_data = "true"
    metric {
      metric_name = "CPUUtilization"
      namespace   = "AWS/EC2"
      period      = 300
      stat        = "Average"
      unit        = "Percent"

      dimensions = {
        AutoScalingGroupName = try(module.ec2.autoscaling_group_name, aws_autoscaling_group.app.name)
      }
    }
  }

  alarm_actions = var.enable_sns_alerts ? [aws_sns_topic.alerts[0].arn] : []
  tags         = var.tags
}

# CloudWatch Log Metric Filters
resource "aws_cloudwatch_log_metric_filter" "error_count" {
  count = var.enable_log_metrics ? 1 : 0

  name           = "${var.project_name}-error-count-${var.environment}"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "[timestamp, level=ERROR, ...]"

  metric_transformation {
    name      = "ErrorCount"
    namespace = "${var.project_name}/${var.environment}"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "warning_count" {
  count = var.enable_log_metrics ? 1 : 0

  name           = "${var.project_name}-warning-count-${var.environment}"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "[timestamp, level=WARNING, ...]"

  metric_transformation {
    name      = "WarningCount"
    namespace = "${var.project_name}/${var.environment}"
    value     = "1"
  }
}

