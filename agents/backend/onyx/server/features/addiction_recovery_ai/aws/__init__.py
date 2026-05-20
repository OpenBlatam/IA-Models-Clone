"""
AWS integration module for Addiction Recovery AI
"""

from .lambda_handler import lambda_handler, get_app, get_handler
from .aws_services import (
    DynamoDBService,
    S3Service,
    CloudWatchService,
    SNSService,
    SQSService,
    SecretsManagerService,
    ParameterStoreService
)
from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from .retry_handler import retry_with_backoff

__all__ = [
    "lambda_handler",
    "get_app",
    "get_handler",
    "DynamoDBService",
    "S3Service",
    "CloudWatchService",
    "SNSService",
    "SQSService",
    "SecretsManagerService",
    "ParameterStoreService",
    "CircuitBreaker",
    "CircuitBreakerError",
    "retry_with_backoff"
]















