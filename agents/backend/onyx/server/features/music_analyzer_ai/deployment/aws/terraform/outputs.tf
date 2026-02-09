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

output "api_gateway_id" {
  description = "API Gateway ID"
  value       = aws_apigatewayv2_api.music_analyzer_api.id
}

output "lambda_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.lambda.name
}




