"""
AWS Configuration for Addiction Recovery AI
Optimized for serverless deployment (Lambda, API Gateway, DynamoDB, etc.)
"""

import os
from typing import Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class AWSSettings(BaseSettings):
    """AWS-specific configuration"""
    
    # AWS Region
    aws_region: str = Field(
        default=os.getenv("AWS_REGION", "us-east-1"),
        description="AWS region"
    )
    
    # Lambda Configuration
    lambda_function_name: str = Field(
        default=os.getenv("LAMBDA_FUNCTION_NAME", "addiction-recovery-ai"),
        description="Lambda function name"
    )
    lambda_timeout: int = Field(
        default=int(os.getenv("LAMBDA_TIMEOUT", "300")),
        description="Lambda timeout in seconds"
    )
    lambda_memory: int = Field(
        default=int(os.getenv("LAMBDA_MEMORY", "512")),
        description="Lambda memory in MB"
    )
    
    # DynamoDB Configuration
    dynamodb_table_name: str = Field(
        default=os.getenv("DYNAMODB_TABLE_NAME", "addiction-recovery-users"),
        description="DynamoDB table name"
    )
    dynamodb_endpoint_url: Optional[str] = Field(
        default=os.getenv("DYNAMODB_ENDPOINT_URL"),
        description="DynamoDB endpoint URL (for local testing)"
    )
    
    # S3 Configuration
    s3_bucket_name: str = Field(
        default=os.getenv("S3_BUCKET_NAME", "addiction-recovery-data"),
        description="S3 bucket name for data storage"
    )
    s3_endpoint_url: Optional[str] = Field(
        default=os.getenv("S3_ENDPOINT_URL"),
        description="S3 endpoint URL (for local testing)"
    )
    
    # CloudWatch Configuration
    cloudwatch_log_group: str = Field(
        default=os.getenv("CLOUDWATCH_LOG_GROUP", "/aws/lambda/addiction-recovery-ai"),
        description="CloudWatch log group name"
    )
    cloudwatch_metrics_namespace: str = Field(
        default=os.getenv("CLOUDWATCH_METRICS_NAMESPACE", "AddictionRecoveryAI"),
        description="CloudWatch metrics namespace"
    )
    
    # X-Ray Tracing
    enable_xray: bool = Field(
        default=os.getenv("ENABLE_XRAY", "true").lower() == "true",
        description="Enable AWS X-Ray tracing"
    )
    
    # API Gateway
    api_gateway_stage: str = Field(
        default=os.getenv("API_GATEWAY_STAGE", "prod"),
        description="API Gateway stage"
    )
    api_gateway_id: Optional[str] = Field(
        default=os.getenv("API_GATEWAY_ID"),
        description="API Gateway ID"
    )
    
    # ElastiCache/Redis (if using managed Redis)
    redis_endpoint: Optional[str] = Field(
        default=os.getenv("REDIS_ENDPOINT"),
        description="Redis/ElastiCache endpoint"
    )
    redis_port: int = Field(
        default=int(os.getenv("REDIS_PORT", "6379")),
        description="Redis port"
    )
    
    # SQS Configuration (for async processing)
    sqs_queue_url: Optional[str] = Field(
        default=os.getenv("SQS_QUEUE_URL"),
        description="SQS queue URL for background tasks"
    )
    
    # SNS Configuration (for notifications)
    sns_topic_arn: Optional[str] = Field(
        default=os.getenv("SNS_TOPIC_ARN"),
        description="SNS topic ARN for notifications"
    )
    
    # Secrets Manager
    secrets_manager_secret_name: Optional[str] = Field(
        default=os.getenv("SECRETS_MANAGER_SECRET_NAME"),
        description="AWS Secrets Manager secret name"
    )
    
    # Parameter Store
    parameter_store_prefix: str = Field(
        default=os.getenv("PARAMETER_STORE_PREFIX", "/addiction-recovery-ai"),
        description="SSM Parameter Store prefix"
    )
    
    # Cold Start Optimization
    preload_models: bool = Field(
        default=os.getenv("PRELOAD_MODELS", "true").lower() == "true",
        description="Preload AI models on cold start"
    )
    
    # Connection Pooling
    connection_pool_size: int = Field(
        default=int(os.getenv("CONNECTION_POOL_SIZE", "5")),
        description="Connection pool size for AWS services"
    )
    
    # Retry Configuration
    max_retries: int = Field(
        default=int(os.getenv("MAX_RETRIES", "3")),
        description="Maximum retry attempts"
    )
    retry_backoff_factor: float = Field(
        default=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
        description="Retry backoff factor"
    )
    
    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(
        default=int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")),
        description="Circuit breaker failure threshold"
    )
    circuit_breaker_timeout: int = Field(
        default=int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60")),
        description="Circuit breaker timeout in seconds"
    )
    
    # Environment
    environment: str = Field(
        default=os.getenv("ENVIRONMENT", "production"),
        description="Deployment environment"
    )
    is_lambda: bool = Field(
        default=os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None,
        description="Running in Lambda environment"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global AWS settings instance
_aws_settings: Optional[AWSSettings] = None


def get_aws_settings() -> AWSSettings:
    """Get AWS settings (singleton)"""
    global _aws_settings
    if _aws_settings is None:
        _aws_settings = AWSSettings()
    return _aws_settings















