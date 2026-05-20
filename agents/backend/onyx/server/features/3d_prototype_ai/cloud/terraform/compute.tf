# Compute resources
# EC2 Instance configuration

# EC2 Instance
resource "aws_instance" "app" {
  ami                    = local.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.app.id]
  subnet_id              = local.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  # User data for initialization
  user_data = base64encode(templatefile(local.user_data_template, {
    app_port     = var.app_port
    app_host     = var.app_host
    project_name = var.project_name
  }))

  # Root block device
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.volume_size
    encrypted             = true
    delete_on_termination = true
    iops                  = var.volume_iops > 0 ? var.volume_iops : null
    throughput            = var.volume_throughput > 0 ? var.volume_throughput : null
  }

  # Additional EBS volumes
  dynamic "ebs_block_device" {
    for_each = var.additional_volumes
    content {
      device_name = ebs_block_device.value.device_name
      volume_type = ebs_block_device.value.volume_type
      volume_size = ebs_block_device.value.volume_size
      encrypted   = ebs_block_device.value.encrypted
      iops        = ebs_block_device.value.iops > 0 ? ebs_block_device.value.iops : null
      throughput  = ebs_block_device.value.throughput > 0 ? ebs_block_device.value.throughput : null
    }
  }

  # Metadata options for security
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  # Enable detailed monitoring
  monitoring = var.enable_detailed_monitoring

  # Disable API termination protection (can be enabled via variable)
  disable_api_termination = var.enable_termination_protection

  tags = merge(
    {
      Name = "${local.name_prefix}-instance"
    },
    local.common_tags
  )

  lifecycle {
    create_before_destroy = true
    ignore_changes = [
      user_data
    ]
  }
}

# Elastic IP
resource "aws_eip" "app" {
  count  = var.allocate_elastic_ip ? 1 : 0
  domain = "vpc"

  instance                  = aws_instance.app.id
  associate_with_private_ip = aws_instance.app.private_ip

  tags = merge(
    {
      Name = "${local.name_prefix}-eip"
    },
    local.common_tags
  )

  depends_on = [aws_internet_gateway.main]
}

# CloudWatch Alarm for CPU
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  count               = var.enable_cloudwatch_alarms ? 1 : 0
  alarm_name          = "${local.name_prefix}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = var.cpu_threshold
  alarm_description   = "This metric monitors EC2 CPU utilization"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    InstanceId = aws_instance.app.id
  }

  tags = local.common_tags
}

# CloudWatch Alarm for Memory (requires CloudWatch agent)
resource "aws_cloudwatch_metric_alarm" "memory_high" {
  count               = var.enable_cloudwatch_alarms && var.enable_memory_alarm ? 1 : 0
  alarm_name          = "${local.name_prefix}-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "mem_used_percent"
  namespace           = "CWAgent"
  period              = 300
  statistic           = "Average"
  threshold           = var.memory_threshold
  alarm_description   = "This metric monitors EC2 memory utilization"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    InstanceId = aws_instance.app.id
  }

  tags = local.common_tags
}

