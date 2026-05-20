output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "load_balancer_dns" {
  description = "Application Load Balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Application Load Balancer zone ID"
  value       = aws_lb.main.zone_id
}

output "ec2_instance_ids" {
  description = "EC2 instance IDs"
  value       = try(aws_instance.app[*].id, aws_autoscaling_group.app[*].id, [])
}

output "ec2_instance_public_ips" {
  description = "EC2 instance public IPs"
  value       = try(aws_instance.app[*].public_ip, [])
}

output "ec2_instance_private_ips" {
  description = "EC2 instance private IPs"
  value       = try(aws_instance.app[*].private_ip, [])
}

output "security_group_id" {
  description = "Security group ID for EC2 instances"
  value       = aws_security_group.app.id
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint (if enabled)"
  value       = try(aws_elasticache_replication_group.redis[0].configuration_endpoint_address, aws_elasticache_cluster.redis[0].cache_nodes[0].address, "")
}

output "redis_url" {
  description = "Redis connection URL"
  value       = var.enable_elasticache ? "redis://${try(aws_elasticache_replication_group.redis[0].configuration_endpoint_address, aws_elasticache_cluster.redis[0].cache_nodes[0].address)}:6379" : "redis://localhost:6379"
}

output "application_url" {
  description = "Application URL"
  value       = "http://${aws_lb.main.dns_name}"
}

output "health_check_url" {
  description = "Health check URL"
  value       = "http://${aws_lb.main.dns_name}${var.health_check_path}"
}

output "ssh_command" {
  description = "SSH command to connect to instances"
  value       = "ssh -i ${var.key_name}.pem ubuntu@${try(aws_instance.app[0].public_ip, module.ec2.autoscaling_group_name, \"<instance-ip>\")}"
}

output "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  value       = try(aws_s3_bucket.terraform_state.id, "")
}

output "terraform_state_lock_table" {
  description = "DynamoDB table for Terraform state locking"
  value       = try(aws_dynamodb_table.terraform_state_lock.id, "")
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = try(aws_sns_topic.alerts[0].arn, "")
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = try(aws_cloudwatch_log_group.app.name, "")
}

