"""
Settings organized into logical groups for better maintainability.
"""

from typing import Optional
from pydantic import Field


class AppSettings:
    """Application-level settings."""
    app_name: str = "Suno Clone AI"
    app_version: str = "1.0.0"
    debug: bool = False


class APISettings:
    """API server settings."""
    api_host: str = "0.0.0.0"
    api_port: int = 8020


class OpenAISettings:
    """OpenAI integration settings."""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"


class MusicGenerationSettings:
    """Music generation model settings."""
    music_model: str = "facebook/musicgen-medium"
    use_gpu: bool = True
    max_audio_length: int = 300
    default_duration: int = 30
    sample_rate: int = 32000
    top_k: int = 250
    top_p: float = 0.0
    temperature: float = 1.0
    cfg_coef: float = 3.0


class DatabaseSettings:
    """Database configuration."""
    database_url: str = "sqlite:///./suno_clone.db"


class SecuritySettings:
    """Security and authentication settings."""
    secret_key: str = "your-secret-key-change-in-production"
    jwt_secret_key: Optional[str] = None
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    enable_auth: bool = False


class RateLimitSettings:
    """Rate limiting configuration."""
    rate_limit_requests: int = 50
    rate_limit_window: int = 60


class StorageSettings:
    """File storage settings."""
    audio_storage_path: str = "./storage/audio"
    max_file_size_mb: int = 50


class CacheSettings:
    """Caching configuration."""
    redis_url: Optional[str] = None
    cache_ttl: int = 3600


class AWSSettings:
    """AWS service configuration."""
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    dynamodb_table_name: Optional[str] = None
    dynamodb_endpoint_url: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    sqs_queue_url: Optional[str] = None
    sqs_queue_name: Optional[str] = None
    cloudwatch_log_group: str = "suno-clone-ai"
    cloudwatch_log_stream: Optional[str] = None
    cloudwatch_metrics_namespace: str = "SunoCloneAI"
    is_lambda: bool = False
    lambda_timeout: int = 900


class ObservabilitySettings:
    """Observability and monitoring settings."""
    otlp_endpoint: Optional[str] = None
    enable_tracing: bool = True


class ResilienceSettings:
    """Resilience patterns configuration."""
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    retry_max_attempts: int = 3
    retry_initial_wait: float = 1.0
    retry_max_wait: float = 60.0


class InfrastructureSettings:
    """Infrastructure and deployment settings."""
    enable_service_mesh: bool = True
    service_mesh_type: str = "istio"
    api_gateway_type: str = "aws"
    enable_api_gateway: bool = True


class PerformanceSettings:
    """Performance optimization settings."""
    enable_response_caching: bool = True
    response_cache_ttl: int = 300
    enable_compression: bool = True
    compression_min_size: int = 500
    compression_level: int = 6
    enable_query_cache: bool = True
    query_cache_size: int = 128
    enable_jit: bool = False
    enable_aggressive_gc: bool = False
    batch_size: int = 100
    max_io_workers: int = 10
    max_concurrent_requests: int = 50
    max_concurrent_tasks: int = 100


class EventBusSettings:
    """Event bus configuration."""
    enable_event_bus: bool = True
    event_bus_backend: str = "memory"

