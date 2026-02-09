"""
AWS Integration Module para Suno Clone AI
"""

from aws.lambda_handler import lambda_handler, sqs_handler, s3_handler
from aws.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    circuit_breaker,
    get_circuit_breaker
)
from aws.exceptions import (
    CircuitBreakerOpenError,
    AWSConnectionError,
    DynamoDBError,
    S3Error,
    SQSError,
    CloudWatchError
)

__all__ = [
    "lambda_handler",
    "sqs_handler",
    "s3_handler",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "circuit_breaker",
    "get_circuit_breaker",
    "CircuitBreakerOpenError",
    "AWSConnectionError",
    "DynamoDBError",
    "S3Error",
    "SQSError",
    "CloudWatchError",
]















