"""
Excepciones personalizadas para AWS y circuit breakers
"""


class CircuitBreakerOpenError(Exception):
    """Excepción cuando el circuit breaker está abierto"""
    pass


class AWSConnectionError(Exception):
    """Error de conexión con servicios AWS"""
    pass


class DynamoDBError(Exception):
    """Error al interactuar con DynamoDB"""
    pass


class S3Error(Exception):
    """Error al interactuar con S3"""
    pass


class SQSError(Exception):
    """Error al interactuar con SQS"""
    pass


class CloudWatchError(Exception):
    """Error al interactuar con CloudWatch"""
    pass















