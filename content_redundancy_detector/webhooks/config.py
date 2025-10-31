from pydantic import BaseSettings, Field
from typing import Optional


class WebhookSettings(BaseSettings):
    WEBHOOK_QUEUE_SIZE: int = Field(default=5000)
    WEBHOOK_MAX_WORKERS: int = Field(default=8)
    WEBHOOK_BATCH_SIZE: int = Field(default=50)
    WEBHOOK_BATCH_WAIT_SECS: float = Field(default=0.25)
    WEBHOOK_HMAC_SECRET: Optional[str] = Field(default=None)
    WEBHOOK_RETRY_MAX_ATTEMPTS: int = Field(default=5)
    WEBHOOK_RETRY_INITIAL_DELAY: float = Field(default=0.5)
    WEBHOOK_RETRY_MAX_DELAY: float = Field(default=10.0)
    WEBHOOK_IDEMPOTENCY_TTL_SECS: float = Field(default=300.0)
    WEBHOOK_BACKPRESSURE_THRESHOLD: float = Field(default=0.9)
    WEBHOOK_RETRY_AFTER_SECS: float = Field(default=5.0)
    WEBHOOK_HTTP_POOL_SIZE: int = Field(default=100)
    WEBHOOK_HTTP_TIMEOUT: float = Field(default=30.0)
    WEBHOOK_PREWARM_WORKERS: bool = Field(default=True)
    # Security
    WEBHOOK_REQUIRE_TIMESTAMP: bool = Field(default=True)
    WEBHOOK_HMAC_WINDOW_SECS: int = Field(default=300)  # 5 minutes
    WEBHOOK_CLOCK_SKEW_SECS: int = Field(default=60)

    class Config:
        env_file = ".env"


settings = WebhookSettings()

"""
Webhooks Configuration - Environment-based configuration
"""

import os
from typing import Optional, Dict, Any


class WebhookConfig:
    """Configuration for webhook system"""
    
    # Storage configuration
    STORAGE_TYPE: str = os.getenv("WEBHOOK_STORAGE_TYPE", "auto")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Performance configuration
    MAX_WORKERS: Optional[int] = None
    MAX_QUEUE_SIZE: int = int(os.getenv("WEBHOOK_MAX_QUEUE_SIZE", "1000"))
    
    # Observability configuration
    ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "true").lower() == "true"
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    OTLP_ENDPOINT: Optional[str] = os.getenv("OTLP_ENDPOINT")
    
    # Serverless detection
    AWS_LAMBDA_FUNCTION_NAME: Optional[str] = os.getenv("AWS_LAMBDA_FUNCTION_NAME")
    FUNCTION_APP: Optional[str] = os.getenv("FUNCTION_APP")
    FUNCTION_NAME: Optional[str] = os.getenv("FUNCTION_NAME")
    
    # Webhook delivery settings
    DEFAULT_TIMEOUT: int = int(os.getenv("WEBHOOK_DEFAULT_TIMEOUT", "30"))
    DEFAULT_RETRY_COUNT: int = int(os.getenv("WEBHOOK_DEFAULT_RETRY_COUNT", "3"))
    MAX_RETRY_DELAY: int = int(os.getenv("WEBHOOK_MAX_RETRY_DELAY", "300"))  # 5 minutes
    
    # Circuit breaker settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(
        os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
    )
    CIRCUIT_BREAKER_TIMEOUT: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
    
    @classmethod
    def is_serverless(cls) -> bool:
        """Check if running in serverless environment"""
        return bool(
            cls.AWS_LAMBDA_FUNCTION_NAME or 
            cls.FUNCTION_APP or 
            cls.FUNCTION_NAME
        )
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        config = {
            "url": cls.REDIS_URL,
            "db": cls.REDIS_DB,
        }
        if cls.REDIS_PASSWORD:
            config["password"] = cls.REDIS_PASSWORD
        return config
    
    @classmethod
    def detect_max_workers(cls) -> int:
        """Auto-detect optimal number of workers"""
        if cls.MAX_WORKERS:
            return cls.MAX_WORKERS
        
        if cls.is_serverless():
            # Minimize workers for serverless
            cpu_count = os.cpu_count() or 1
            return min(2, cpu_count)
        
        # Container/K8s: use available CPUs
        cpu_count = os.cpu_count() or 4
        return min(cpu_count, 10)
    
    @classmethod
    def get_http_client_config(cls) -> Dict[str, Any]:
        """Get HTTP client configuration"""
        max_connections = 5 if cls.is_serverless() else 100
        
        return {
            "timeout": 30.0,
            "connect_timeout": 10.0,
            "max_keepalive_connections": max_connections,
            "max_connections": max_connections,
            "follow_redirects": True,
        }

