variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "robot-movement-ai"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
}

variable "robot_ip" {
  description = "Robot IP address"
  type        = string
  default     = "192.168.1.100"
}

variable "enable_redis" {
  description = "Enable ElastiCache Redis"
  type        = bool
  default     = true
}

variable "enable_kafka" {
  description = "Enable MSK (Managed Streaming for Kafka)"
  type        = bool
  default     = false
}

variable "enable_celery" {
  description = "Enable Celery workers"
  type        = bool
  default     = true
}

variable "enable_api_gateway" {
  description = "Enable API Gateway"
  type        = bool
  default     = false
}

variable "enable_waf" {
  description = "Enable WAF for DDoS protection"
  type        = bool
  default     = true
}

