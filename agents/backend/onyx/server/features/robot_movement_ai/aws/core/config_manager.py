"""
Configuration Manager
=====================

Manages configuration in a modular way.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MiddlewareConfig:
    """Middleware configuration."""
    enable_tracing: bool = True
    enable_rate_limiting: bool = True
    enable_circuit_breaker: bool = True
    enable_caching: bool = True
    enable_logging: bool = True
    enable_security_headers: bool = True
    
    # Tracing
    otlp_endpoint: str = "http://localhost:4317"
    otlp_insecure: bool = True
    
    # Rate limiting
    redis_url: Optional[str] = None
    rate_limit_per_minute: int = 60
    
    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout: int = 60
    
    # Caching
    cache_ttl: int = 300
    
    @classmethod
    def from_env(cls) -> "MiddlewareConfig":
        """Create from environment variables."""
        return cls(
            enable_tracing=os.getenv("ENABLE_TRACING", "true").lower() == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            enable_circuit_breaker=os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower() == "true",
            enable_caching=os.getenv("ENABLE_REDIS_CACHE", "true").lower() == "true",
            enable_logging=True,
            enable_security_headers=True,
            otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
            otlp_insecure=os.getenv("OTLP_INSECURE", "true").lower() == "true",
            redis_url=os.getenv("REDIS_URL"),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            failure_threshold=int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")),
            recovery_timeout=int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60")),
            cache_ttl=int(os.getenv("CACHE_TTL", "300")),
        )


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_prometheus: bool = True
    enable_cloudwatch: bool = True
    enable_sentry: bool = False
    
    prometheus_port: int = 9090
    sentry_dsn: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create from environment variables."""
        return cls(
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            enable_cloudwatch=os.getenv("ENABLE_CLOUDWATCH", "true").lower() == "true",
            enable_sentry=os.getenv("ENABLE_SENTRY", "false").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            sentry_dsn=os.getenv("SENTRY_DSN"),
        )


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_oauth2: bool = True
    enable_jwt: bool = True
    enable_rate_limiting: bool = True
    
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Create from environment variables."""
        return cls(
            enable_oauth2=os.getenv("ENABLE_OAUTH2", "true").lower() == "true",
            enable_jwt=os.getenv("ENABLE_JWT", "true").lower() == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", "change-me-in-production"),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")),
        )


@dataclass
class MessagingConfig:
    """Messaging configuration."""
    enable_kafka: bool = False
    enable_rabbitmq: bool = False
    enable_sqs: bool = False
    
    kafka_bootstrap_servers: Optional[str] = None
    kafka_topic_prefix: str = "robot-movement-ai"
    
    rabbitmq_url: Optional[str] = None
    
    sqs_queue_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "MessagingConfig":
        """Create from environment variables."""
        return cls(
            enable_kafka=os.getenv("ENABLE_KAFKA", "false").lower() == "true",
            enable_rabbitmq=os.getenv("ENABLE_RABBITMQ", "false").lower() == "true",
            enable_sqs=os.getenv("ENABLE_SQS", "false").lower() == "true",
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            kafka_topic_prefix=os.getenv("KAFKA_TOPIC_PREFIX", "robot-movement-ai"),
            rabbitmq_url=os.getenv("RABBITMQ_URL"),
            sqs_queue_url=os.getenv("SQS_QUEUE_URL"),
        )


@dataclass
class WorkerConfig:
    """Worker configuration."""
    enable_celery: bool = True
    enable_rq: bool = False
    
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    
    rq_redis_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Create from environment variables."""
        return cls(
            enable_celery=os.getenv("ENABLE_CELERY", "true").lower() == "true",
            enable_rq=os.getenv("ENABLE_RQ", "false").lower() == "true",
            celery_broker_url=os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL"),
            celery_result_backend=os.getenv("CELERY_RESULT_BACKEND") or os.getenv("REDIS_RESULT_BACKEND"),
            rq_redis_url=os.getenv("RQ_REDIS_URL") or os.getenv("REDIS_URL"),
        )


@dataclass
class CacheConfig:
    """Cache configuration."""
    enable_redis: bool = True
    enable_memcached: bool = False
    
    redis_url: Optional[str] = None
    memcached_servers: Optional[str] = None
    
    default_ttl: int = 300
    
    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create from environment variables."""
        return cls(
            enable_redis=os.getenv("ENABLE_REDIS", "true").lower() == "true",
            enable_memcached=os.getenv("ENABLE_MEMCACHED", "false").lower() == "true",
            redis_url=os.getenv("REDIS_URL"),
            memcached_servers=os.getenv("MEMCACHED_SERVERS"),
            default_ttl=int(os.getenv("CACHE_TTL", "300")),
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig.from_env)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig.from_env)
    security: SecurityConfig = field(default_factory=SecurityConfig.from_env)
    messaging: MessagingConfig = field(default_factory=MessagingConfig.from_env)
    worker: WorkerConfig = field(default_factory=WorkerConfig.from_env)
    cache: CacheConfig = field(default_factory=CacheConfig.from_env)
    
    app_name: str = "robot-movement-ai"
    app_version: str = "1.0.0"
    environment: str = "production"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create from environment variables."""
        return cls(
            middleware=MiddlewareConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            security=SecurityConfig.from_env(),
            messaging=MessagingConfig.from_env(),
            worker=WorkerConfig.from_env(),
            cache=CacheConfig.from_env(),
            app_name=os.getenv("APP_NAME", "robot-movement-ai"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "production"),
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "AppConfig":
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            middleware=MiddlewareConfig(**data.get("middleware", {})),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
            security=SecurityConfig(**data.get("security", {})),
            messaging=MessagingConfig(**data.get("messaging", {})),
            worker=WorkerConfig(**data.get("worker", {})),
            cache=CacheConfig(**data.get("cache", {})),
            app_name=data.get("app_name", "robot-movement-ai"),
            app_version=data.get("app_version", "1.0.0"),
            environment=data.get("environment", "production"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "middleware": self.middleware.__dict__,
            "monitoring": self.monitoring.__dict__,
            "security": self.security.__dict__,
            "messaging": self.messaging.__dict__,
            "worker": self.worker.__dict__,
            "cache": self.cache.__dict__,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
        }















