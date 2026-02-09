"""
Configuración del sistema Suno Clone AI

Refactorizado para usar estructura agrupada y mejor organización.
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Configuración de la aplicación.
    
    Organizado en grupos lógicos para mejor mantenibilidad.
    Mantiene compatibilidad con código existente mediante propiedades.
    """
    
    # Aplicación
    app_name: str = os.getenv("APP_NAME", "Suno Clone AI")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8020"))
    
    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Music Generation Models
    music_model: str = os.getenv("MUSIC_MODEL", "facebook/musicgen-medium")
    use_gpu: bool = os.getenv("USE_GPU", "True").lower() == "true"
    max_audio_length: int = int(os.getenv("MAX_AUDIO_LENGTH", "300"))
    default_duration: int = int(os.getenv("DEFAULT_DURATION", "30"))
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "32000"))
    top_k: int = int(os.getenv("TOP_K", "250"))
    top_p: float = float(os.getenv("TOP_P", "0.0"))
    temperature: float = float(os.getenv("TEMPERATURE", "1.0"))
    cfg_coef: float = float(os.getenv("CFG_COEF", "3.0"))
    
    # Base de datos
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./suno_clone.db")
    
    # Seguridad
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    jwt_secret_key: str = os.getenv(
        "JWT_SECRET_KEY", 
        os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    enable_auth: bool = os.getenv("ENABLE_AUTH", "False").lower() == "true"
    
    # Rate limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "50"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # Storage
    audio_storage_path: str = os.getenv("AUDIO_STORAGE_PATH", "./storage/audio")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # Cache
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # AWS Configuration
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # DynamoDB
    dynamodb_table_name: Optional[str] = os.getenv("DYNAMODB_TABLE_NAME")
    dynamodb_endpoint_url: Optional[str] = os.getenv("DYNAMODB_ENDPOINT_URL")
    
    # S3
    s3_bucket_name: Optional[str] = os.getenv("S3_BUCKET_NAME")
    s3_endpoint_url: Optional[str] = os.getenv("S3_ENDPOINT_URL")
    
    # SQS
    sqs_queue_url: Optional[str] = os.getenv("SQS_QUEUE_URL")
    sqs_queue_name: Optional[str] = os.getenv("SQS_QUEUE_NAME")
    
    # CloudWatch
    cloudwatch_log_group: str = os.getenv("CLOUDWATCH_LOG_GROUP", "suno-clone-ai")
    cloudwatch_log_stream: Optional[str] = os.getenv("CLOUDWATCH_LOG_STREAM")
    cloudwatch_metrics_namespace: str = os.getenv("CLOUDWATCH_METRICS_NAMESPACE", "SunoCloneAI")
    
    # Lambda/Serverless
    is_lambda: bool = os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None
    lambda_timeout: int = int(os.getenv("LAMBDA_TIMEOUT", "900"))
    
    # OpenTelemetry
    otlp_endpoint: Optional[str] = os.getenv("OTLP_ENDPOINT")
    enable_tracing: bool = os.getenv("ENABLE_TRACING", "True").lower() == "true"
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = os.getenv("CIRCUIT_BREAKER_ENABLED", "True").lower() == "true"
    circuit_breaker_failure_threshold: int = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    circuit_breaker_timeout: float = float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60.0"))
    
    # Retry Configuration
    retry_max_attempts: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    retry_initial_wait: float = float(os.getenv("RETRY_INITIAL_WAIT", "1.0"))
    retry_max_wait: float = float(os.getenv("RETRY_MAX_WAIT", "60.0"))
    
    # Service Mesh
    enable_service_mesh: bool = os.getenv("ENABLE_SERVICE_MESH", "True").lower() == "true"
    service_mesh_type: str = os.getenv("SERVICE_MESH_TYPE", "istio")
    
    # API Gateway
    api_gateway_type: str = os.getenv("API_GATEWAY_TYPE", "aws")
    enable_api_gateway: bool = os.getenv("ENABLE_API_GATEWAY", "True").lower() == "true"
    
    # Performance
    enable_response_caching: bool = os.getenv("ENABLE_RESPONSE_CACHING", "True").lower() == "true"
    response_cache_ttl: int = int(os.getenv("RESPONSE_CACHE_TTL", "300"))
    enable_compression: bool = os.getenv("ENABLE_COMPRESSION", "True").lower() == "true"
    compression_min_size: int = int(os.getenv("COMPRESSION_MIN_SIZE", "500"))
    compression_level: int = int(os.getenv("COMPRESSION_LEVEL", "6"))
    enable_query_cache: bool = os.getenv("ENABLE_QUERY_CACHE", "True").lower() == "true"
    query_cache_size: int = int(os.getenv("QUERY_CACHE_SIZE", "128"))
    
    # Advanced Optimizations
    enable_jit: bool = os.getenv("ENABLE_JIT", "False").lower() == "true"
    enable_aggressive_gc: bool = os.getenv("ENABLE_AGGRESSIVE_GC", "False").lower() == "true"
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
    max_io_workers: int = int(os.getenv("MAX_IO_WORKERS", "10"))
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "100"))
    
    # Event Bus
    enable_event_bus: bool = os.getenv("ENABLE_EVENT_BUS", "True").lower() == "true"
    event_bus_backend: str = os.getenv("EVENT_BUS_BACKEND", "memory")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

