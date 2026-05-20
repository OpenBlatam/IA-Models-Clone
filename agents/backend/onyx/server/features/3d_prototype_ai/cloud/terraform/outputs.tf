# Instance outputs
output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.app.id
}

output "instance_public_ip" {
  description = "EC2 instance public IP"
  value       = var.allocate_elastic_ip ? aws_eip.app[0].public_ip : aws_instance.app.public_ip
}

output "instance_private_ip" {
  description = "EC2 instance private IP"
  value       = aws_instance.app.private_ip
}

output "instance_public_dns" {
  description = "EC2 instance public DNS"
  value       = aws_instance.app.public_dns
}

output "elastic_ip" {
  description = "Elastic IP address (if allocated)"
  value       = var.allocate_elastic_ip ? aws_eip.app[0].public_ip : null
  sensitive   = false
}

# Application URLs
output "application_url" {
  description = "Application HTTP URL"
  value       = local.app_url_http
}

output "application_url_https" {
  description = "Application HTTPS URL (if domain configured)"
  value       = local.app_url_https
}

# Connection information
output "ssh_command" {
  description = "SSH command to connect to instance"
  value       = "ssh -i ${var.key_name}.pem ubuntu@${var.allocate_elastic_ip ? aws_eip.app[0].public_ip : aws_instance.app.public_ip}"
}

# Network information
output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.app.id
}

output "vpc_id" {
  description = "VPC ID"
  value       = local.vpc_id
}

output "subnet_id" {
  description = "Subnet ID"
  value       = local.subnet_id
}

# Storage information
output "s3_backup_bucket" {
  description = "S3 backup bucket name (if enabled)"
  value       = var.enable_s3_backup ? aws_s3_bucket.backup[0].id : null
}

output "s3_backup_bucket_arn" {
  description = "S3 backup bucket ARN (if enabled)"
  value       = var.enable_s3_backup ? aws_s3_bucket.backup[0].arn : null
}

# Monitoring information
output "cloudwatch_alarms" {
  description = "CloudWatch alarm names"
  value = var.enable_cloudwatch_alarms ? [
    aws_cloudwatch_metric_alarm.cpu_high[0].alarm_name,
    var.enable_memory_alarm ? aws_cloudwatch_metric_alarm.memory_high[0].alarm_name : null
  ] : []
}

# Summary
output "deployment_summary" {
  description = "Deployment summary"
  value = {
    instance_id      = aws_instance.app.id
    public_ip        = var.allocate_elastic_ip ? aws_eip.app[0].public_ip : aws_instance.app.public_ip
    application_url  = local.app_url_http
    environment      = var.environment
    region           = var.aws_region
    instance_type    = var.instance_type
  }
}

