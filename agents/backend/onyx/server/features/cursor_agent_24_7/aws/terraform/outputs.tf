# Outputs de Terraform
# =====================

output "ecr_repository_url" {
  description = "URL del repositorio ECR"
  value       = aws_ecr_repository.cursor_agent.repository_url
}

output "alb_dns_name" {
  description = "DNS name del Application Load Balancer"
  value       = aws_lb.cursor_agent.dns_name
}

output "alb_arn" {
  description = "ARN del Application Load Balancer"
  value       = aws_lb.cursor_agent.arn
}

output "dynamodb_state_table" {
  description = "Nombre de la tabla DynamoDB para estado"
  value       = aws_dynamodb_table.agent_state.name
}

output "dynamodb_cache_table" {
  description = "Nombre de la tabla DynamoDB para caché"
  value       = aws_dynamodb_table.agent_cache.name
}

output "redis_endpoint" {
  description = "Endpoint de ElastiCache Redis"
  value       = var.enable_redis ? aws_elasticache_replication_group.redis.configuration_endpoint_address : null
}

output "cloudwatch_log_group" {
  description = "Nombre del grupo de logs de CloudWatch"
  value       = aws_cloudwatch_log_group.cursor_agent.name
}

output "ecs_cluster_name" {
  description = "Nombre del cluster ECS"
  value       = aws_ecs_cluster.cursor_agent.name
}

output "ecs_service_name" {
  description = "Nombre del servicio ECS"
  value       = aws_ecs_service.cursor_agent.name
}




