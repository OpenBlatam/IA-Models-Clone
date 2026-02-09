terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    # Configure backend in terraform.tfvars
    bucket = "music-analyzer-terraform-state"
    key    = "music-analyzer-ai/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Music Analyzer AI"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Variables
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

# Lambda Function
resource "aws_lambda_function" "music_analyzer" {
  function_name = "music-analyzer-ai-${var.environment}"
  runtime       = "python3.11"
  handler       = "deployment.aws.lambda_handler.lambda_handler"
  memory_size   = var.lambda_memory_size
  timeout       = var.lambda_timeout
  
  # Package code (will be built by CI/CD)
  filename         = "${path.module}/../../package.zip"
  source_code_hash = filebase64sha256("${path.module}/../../package.zip")
  
  role = aws_iam_role.lambda_execution.arn
  
  layers = [aws_lambda_layer_version.dependencies.arn]
  
  environment {
    variables = {
      ENVIRONMENT          = var.environment
      SPOTIFY_CLIENT_ID    = var.spotify_client_id
      SPOTIFY_CLIENT_SECRET = var.spotify_client_secret
      CACHE_ENABLED        = "true"
      LOG_LEVEL            = "INFO"
      CLOUDWATCH_ENABLED   = "true"
      AWS_REGION           = var.aws_region
    }
  }
  
  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }
  
  reserved_concurrent_executions = 100
  
  tracing_config {
    mode = "Active"  # Enable X-Ray tracing
  }
  
  tags = {
    Name = "music-analyzer-ai-${var.environment}"
  }
}

# Lambda Layer for dependencies
resource "aws_lambda_layer_version" "dependencies" {
  filename            = "${path.module}/../../dependencies.zip"
  layer_name          = "music-analyzer-deps-${var.environment}"
  source_code_hash    = filebase64sha256("${path.module}/../../dependencies.zip")
  compatible_runtimes = ["python3.11"]
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_execution" {
  name = "music-analyzer-lambda-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_logs" {
  name = "lambda-cloudwatch-logs-${var.environment}"
  role = aws_iam_role.lambda_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ]
      Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/music-analyzer-ai-${var.environment}*"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_dynamodb" {
  name = "lambda-dynamodb-${var.environment}"
  role = aws_iam_role.lambda_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ]
      Resource = aws_dynamodb_table.cache.arn
    }]
  })
}

# API Gateway V2 (HTTP API)
resource "aws_apigatewayv2_api" "music_analyzer_api" {
  name          = "music-analyzer-api-${var.environment}"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers = ["*"]
    max_age       = 3600
  }
}

# API Gateway Integration
resource "aws_apigatewayv2_integration" "lambda" {
  api_id           = aws_apigatewayv2_api.music_analyzer_api.id
  integration_type = "AWS_PROXY"
  
  integration_uri    = aws_lambda_function.music_analyzer.invoke_arn
  payload_format_version = "2.0"
}

# API Gateway Route
resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.music_analyzer_api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# API Gateway Stage
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.music_analyzer_api.id
  name        = var.environment
  auto_deploy = true
  
  default_route_settings {
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }
  
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
}

# Lambda Permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.music_analyzer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.music_analyzer_api.execution_arn}/*/*"
}

# Dead Letter Queue
resource "aws_sqs_queue" "dlq" {
  name                       = "music-analyzer-dlq-${var.environment}"
  message_retention_seconds  = 1209600  # 14 days
  
  tags = {
    Name = "music-analyzer-dlq-${var.environment}"
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/music-analyzer-ai-${var.environment}"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/music-analyzer-api-${var.environment}"
  retention_in_days = 30
}

# DynamoDB Table for caching
resource "aws_dynamodb_table" "cache" {
  name         = "music-analyzer-cache-${var.environment}"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "key"
  
  attribute {
    name = "key"
    type = "S"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  tags = {
    Name = "music-analyzer-cache-${var.environment}"
  }
}

# Outputs
output "api_url" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_api.music_analyzer_api.api_endpoint
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.music_analyzer.arn
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.music_analyzer.function_name
}

output "cache_table_name" {
  description = "DynamoDB cache table name"
  value       = aws_dynamodb_table.cache.name
}




