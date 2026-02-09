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

variable "lambda_memory_size" {
  description = "Lambda function memory size (MB)"
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Lambda function timeout (seconds)"
  type        = number
  default     = 300
}

variable "spotify_client_id" {
  description = "Spotify API Client ID"
  type        = string
  sensitive   = true
}

variable "spotify_client_secret" {
  description = "Spotify API Client Secret"
  type        = string
  sensitive   = true
}

variable "api_gateway_throttle_burst" {
  description = "API Gateway throttle burst limit"
  type        = number
  default     = 100
}

variable "api_gateway_throttle_rate" {
  description = "API Gateway throttle rate limit"
  type        = number
  default     = 50
}

variable "enable_xray" {
  description = "Enable AWS X-Ray tracing"
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}




