from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
            import json
from typing import Any, List, Dict, Optional
import asyncio
"""
Configuration Management
=======================

Centralized configuration for the copywriting service with environment-based
settings, intelligent defaults, and validation.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = os.getenv("DATABASE_URL", "postgresql://localhost/copywriting")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
    socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
    socket_connect_timeout: float = float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0"))
    retry_on_timeout: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"


@dataclass
class AIConfig:
    """AI service configuration"""
    # OpenRouter settings
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Anthropic settings
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Google settings
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Default models
    default_model: str = os.getenv("DEFAULT_AI_MODEL", "gpt-4")
    fallback_model: str = os.getenv("FALLBACK_AI_MODEL", "gpt-3.5-turbo")
    
    # Generation settings
    max_tokens: int = int(os.getenv("AI_MAX_TOKENS", "2000"))
    temperature: float = float(os.getenv("AI_TEMPERATURE", "0.7"))
    timeout: int = int(os.getenv("AI_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("AI_MAX_RETRIES", "3"))


@dataclass
class CacheConfig:
    """Caching configuration"""
    # Memory cache
    memory_cache_size: int = int(os.getenv("MEMORY_CACHE_SIZE", "1000"))
    memory_cache_ttl: int = int(os.getenv("MEMORY_CACHE_TTL", "3600"))
    
    # Redis cache
    redis_cache_ttl: int = int(os.getenv("REDIS_CACHE_TTL", "86400"))
    redis_cache_prefix: str = os.getenv("REDIS_CACHE_PREFIX", "copywriting:")
    
    # Compression
    enable_compression: bool = os.getenv("CACHE_COMPRESSION", "true").lower() == "true"
    compression_threshold: int = int(os.getenv("COMPRESSION_THRESHOLD", "1024"))
    compression_level: int = int(os.getenv("COMPRESSION_LEVEL", "6"))


@dataclass
class OptimizationConfig:
    """Performance optimization settings"""
    # JIT compilation
    enable_jit: bool = os.getenv("ENABLE_JIT", "true").lower() == "true"
    jit_cache_size: int = int(os.getenv("JIT_CACHE_SIZE", "100"))
    
    # Async settings
    max_workers: int = int(os.getenv("MAX_WORKERS", str(os.cpu_count() or 4)))
    worker_timeout: int = int(os.getenv("WORKER_TIMEOUT", "300"))
    
    # Memory management
    memory_limit_mb: int = int(os.getenv("MEMORY_LIMIT_MB", "1024"))
    gc_threshold: int = int(os.getenv("GC_THRESHOLD", "1000"))
    
    # Batch processing
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "100"))


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    # Metrics
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "8001"))
    
    # Health checks
    health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    
    # Alerts
    enable_alerts: bool = os.getenv("ENABLE_ALERTS", "false").lower() == "true"
    alert_webhook_url: str = os.getenv("ALERT_WEBHOOK_URL", "")


@dataclass
class SecurityConfig:
    """Security configuration"""
    # API Keys
    api_key_header: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    valid_api_keys: List[str] = field(
        default_factory=lambda: os.getenv("VALID_API_KEYS", "").split(",") if os.getenv("VALID_API_KEYS") else []
    )
    
    # Rate limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    rate_limit_burst: int = int(os.getenv("RATE_LIMIT_BURST", "200"))
    
    # CORS
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(",")
    )


@dataclass
class CopywritingConfig:
    """Main configuration class"""
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # General settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Application settings
    app_name: str = os.getenv("APP_NAME", "Copywriting Service")
    app_version: str = os.getenv("APP_VERSION", "2.0.0")
    
    # Supported languages and tones
    supported_languages: List[str] = field(default_factory=lambda: [
        "spanish", "english", "french", "portuguese", "italian", "german",
        "dutch", "russian", "chinese", "japanese", "korean", "arabic",
        "hindi", "turkish", "polish", "swedish", "danish", "norwegian", "finnish"
    ])
    
    supported_tones: List[str] = field(default_factory=lambda: [
        "professional", "casual", "urgent", "inspirational", "conversational",
        "persuasive", "storytelling", "educational", "motivational", "empathetic",
        "confident", "playful", "authoritative", "friendly", "formal", "informal",
        "humorous", "serious", "optimistic", "direct"
    ])
    
    supported_use_cases: List[str] = field(default_factory=lambda: [
        "product_launch", "brand_awareness", "lead_generation", "social_media",
        "email_marketing", "blog_post", "website_copy", "ad_copy", "press_release",
        "newsletter", "sales_page", "landing_page", "case_study", "testimonial",
        "faq", "product_description", "service_description", "company_bio",
        "team_bio", "mission_statement", "value_proposition", "call_to_action",
        "headline", "tagline", "slogan"
    ])
    
    def __post_init__(self) -> Any:
        """Validate configuration after initialization"""
        self._validate_config()
        self._setup_logging()
        
    def _validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate AI configuration
        if not any([
            self.ai.openrouter_api_key,
            self.ai.openai_api_key,
            self.ai.anthropic_api_key,
            self.ai.google_api_key
        ]):
            errors.append("At least one AI API key must be configured")
        
        # Validate database URL
        if not self.database.url or not self.database.url.startswith(("postgresql://", "sqlite://")):
            errors.append("Valid database URL is required")
        
        # Validate Redis URL
        if not self.redis.url or not self.redis.url.startswith("redis://"):
            errors.append("Valid Redis URL is required")
        
        # Validate port ranges
        if not (1024 <= self.port <= 65535):
            errors.append("Port must be between 1024 and 65535")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _setup_logging(self) -> Any:
        """Setup logging configuration"""
        log_level = getattr(logging, self.monitoring.log_level.upper(), logging.INFO)
        
        if self.monitoring.log_format == "json":
            class JsonFormatter(logging.Formatter):
                def format(self, record) -> Any:
                    log_data = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno
                    }
                    if record.exc_info:
                        log_data["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_data)
            
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def get_ai_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific AI provider"""
        configs = {
            "openrouter": {
                "api_key": self.ai.openrouter_api_key,
                "base_url": self.ai.openrouter_base_url,
                "default_model": "openai/gpt-4"
            },
            "openai": {
                "api_key": self.ai.openai_api_key,
                "base_url": self.ai.openai_base_url,
                "default_model": "gpt-4"
            },
            "anthropic": {
                "api_key": self.ai.anthropic_api_key,
                "base_url": "https://api.anthropic.com",
                "default_model": "claude-3-opus-20240229"
            },
            "google": {
                "api_key": self.ai.google_api_key,
                "base_url": "https://generativelanguage.googleapis.com",
                "default_model": "gemini-pro"
            }
        }
        return configs.get(provider, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "supported_languages": self.supported_languages,
            "supported_tones": self.supported_tones,
            "supported_use_cases": self.supported_use_cases
        }


# Global configuration instance
config = CopywritingConfig()


def get_config() -> CopywritingConfig:
    """Get the global configuration instance"""
    return config


def reload_config() -> CopywritingConfig:
    """Reload configuration from environment"""
    global config
    config = CopywritingConfig()
    return config 