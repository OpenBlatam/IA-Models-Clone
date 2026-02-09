from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
import sys
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum
            from dotenv import load_dotenv
                import json
                from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
Enterprise Configuration Management
==================================

Centralized configuration system with:
- Environment-based configuration
- Intelligent defaults and validation
- Security and performance settings
- Multi-environment support
- Configuration hot-reloading
"""


logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class OptimizationLevel(str, Enum):
    """Optimization levels"""
    BASIC = "basic"          # 1-5x performance
    OPTIMIZED = "optimized"  # 5-15x performance  
    ULTRA = "ultra"          # 15-25x performance
    MAXIMUM = "maximum"      # 25-50x performance
    AUTO = "auto"            # Automatic detection


class CacheStrategy(str, Enum):
    """Cache strategies"""
    MEMORY = "memory"
    REDIS = "redis"
    MULTI_LEVEL = "multi_level"
    DISABLED = "disabled"


@dataclass
class AIProviderConfig:
    """AI provider configuration"""
    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Google
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Default settings
    default_model: str = os.getenv("DEFAULT_AI_MODEL", "gpt-4")
    fallback_model: str = os.getenv("FALLBACK_AI_MODEL", "gpt-3.5-turbo")
    max_tokens: int = int(os.getenv("AI_MAX_TOKENS", "2000"))
    temperature: float = float(os.getenv("AI_TEMPERATURE", "0.7"))
    timeout: int = int(os.getenv("AI_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("AI_MAX_RETRIES", "3"))
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers"""
        providers = []
        if self.openrouter_api_key:
            providers.append("openrouter")
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        return providers


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = os.getenv("DATABASE_URL", "postgresql://localhost/copywriting")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    echo: bool = os.getenv("DB_ECHO", "false").lower() == "true"


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
    socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
    socket_connect_timeout: float = float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0"))
    retry_on_timeout: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    health_check_interval: int = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))


@dataclass
class CacheConfig:
    """Cache configuration"""
    strategy: CacheStrategy = CacheStrategy(os.getenv("CACHE_STRATEGY", "multi_level"))
    
    # Memory cache
    memory_size: int = int(os.getenv("MEMORY_CACHE_SIZE", "1000"))
    memory_ttl: int = int(os.getenv("MEMORY_CACHE_TTL", "3600"))
    
    # Redis cache
    redis_ttl: int = int(os.getenv("REDIS_CACHE_TTL", "86400"))
    redis_prefix: str = os.getenv("REDIS_CACHE_PREFIX", "enterprise_copy:")
    
    # Compression
    enable_compression: bool = os.getenv("CACHE_COMPRESSION", "true").lower() == "true"
    compression_threshold: int = int(os.getenv("COMPRESSION_THRESHOLD", "1024"))
    compression_level: int = int(os.getenv("COMPRESSION_LEVEL", "6"))


@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    level: OptimizationLevel = OptimizationLevel(os.getenv("OPTIMIZATION_LEVEL", "auto"))
    
    # JIT compilation
    enable_jit: bool = os.getenv("ENABLE_JIT", "true").lower() == "true"
    jit_cache_size: int = int(os.getenv("JIT_CACHE_SIZE", "100"))
    
    # Async settings
    max_workers: int = int(os.getenv("MAX_WORKERS", str(os.cpu_count() or 4)))
    worker_timeout: int = int(os.getenv("WORKER_TIMEOUT", "300"))
    
    # Memory management
    memory_limit_mb: int = int(os.getenv("MEMORY_LIMIT_MB", "2048"))
    gc_threshold: int = int(os.getenv("GC_THRESHOLD", "1000"))
    
    # Batch processing
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    
    # GPU acceleration
    enable_gpu: bool = os.getenv("ENABLE_GPU", "auto").lower() in ["true", "auto"]
    gpu_memory_fraction: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))


@dataclass
class SecurityConfig:
    """Security configuration"""
    # API Authentication
    api_key_header: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    valid_api_keys: List[str] = field(
        default_factory=lambda: [k.strip() for k in os.getenv("VALID_API_KEYS", "").split(",") if k.strip()]
    )
    
    # Rate limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    rate_limit_burst: int = int(os.getenv("RATE_LIMIT_BURST", "200"))
    rate_limit_strategy: str = os.getenv("RATE_LIMIT_STRATEGY", "sliding_window")
    
    # CORS
    cors_origins: List[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
    )
    cors_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    
    # Encryption
    secret_key: str = os.getenv("SECRET_KEY", "")
    encryption_algorithm: str = os.getenv("ENCRYPTION_ALGORITHM", "HS256")
    
    # SSL/TLS
    ssl_enabled: bool = os.getenv("SSL_ENABLED", "false").lower() == "true"
    ssl_cert_path: str = os.getenv("SSL_CERT_PATH", "")
    ssl_key_path: str = os.getenv("SSL_KEY_PATH", "")


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    # Metrics
    enabled: bool = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "8001"))
    
    # Health checks
    health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    health_check_timeout: int = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    log_file: str = os.getenv("LOG_FILE", "")
    
    # Performance tracking
    performance_sampling_rate: float = float(os.getenv("PERFORMANCE_SAMPLING_RATE", "0.1"))
    
    # Alerts
    enable_alerts: bool = os.getenv("ENABLE_ALERTS", "false").lower() == "true"
    alert_webhook_url: str = os.getenv("ALERT_WEBHOOK_URL", "")
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": float(os.getenv("ALERT_ERROR_RATE_THRESHOLD", "5.0")),
        "response_time": float(os.getenv("ALERT_RESPONSE_TIME_THRESHOLD", "2000.0")),
        "memory_usage": float(os.getenv("ALERT_MEMORY_USAGE_THRESHOLD", "80.0")),
        "cpu_usage": float(os.getenv("ALERT_CPU_USAGE_THRESHOLD", "80.0"))
    })


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "1"))
    reload: bool = os.getenv("RELOAD", "false").lower() == "true"
    access_log: bool = os.getenv("ACCESS_LOG", "true").lower() == "true"
    
    # Timeouts
    keep_alive: int = int(os.getenv("KEEP_ALIVE", "5"))
    timeout_keep_alive: int = int(os.getenv("TIMEOUT_KEEP_ALIVE", "5"))
    timeout_graceful_shutdown: int = int(os.getenv("TIMEOUT_GRACEFUL_SHUTDOWN", "30"))
    
    # Limits
    max_requests: int = int(os.getenv("MAX_REQUESTS", "1000"))
    max_requests_jitter: int = int(os.getenv("MAX_REQUESTS_JITTER", "50"))
    
    # Optimization
    loop: str = os.getenv("LOOP", "auto")  # auto, asyncio, uvloop
    http: str = os.getenv("HTTP", "auto")  # auto, h11, httptools


class EnterpriseConfig:
    """Main enterprise configuration class"""
    
    def __init__(self, env_file: Optional[str] = None):
        
    """__init__ function."""
# Load environment file if specified
        if env_file:
            self._load_env_file(env_file)
        
        # Initialize sub-configurations
        self.environment = Environment(os.getenv("ENVIRONMENT", "production"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Sub-configurations
        self.ai = AIProviderConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.cache = CacheConfig()
        self.optimization = OptimizationConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.server = ServerConfig()
        
        # Application metadata
        self.app_name = os.getenv("APP_NAME", "Enterprise Copywriting Service")
        self.app_version = os.getenv("APP_VERSION", "3.0.0")
        self.app_description = os.getenv("APP_DESCRIPTION", "Enterprise-grade copywriting service")
        
        # Content settings
        self.supported_languages = [
            "spanish", "english", "french", "portuguese", "italian", "german",
            "dutch", "russian", "chinese", "japanese", "korean", "arabic",
            "hindi", "turkish", "polish", "swedish", "danish", "norwegian", "finnish"
        ]
        
        self.supported_tones = [
            "professional", "casual", "urgent", "inspirational", "conversational",
            "persuasive", "storytelling", "educational", "motivational", "empathetic",
            "confident", "playful", "authoritative", "friendly", "formal", "informal",
            "humorous", "serious", "optimistic", "direct"
        ]
        
        self.supported_use_cases = [
            "product_launch", "brand_awareness", "lead_generation", "social_media",
            "email_marketing", "blog_post", "website_copy", "ad_copy", "press_release",
            "newsletter", "sales_page", "landing_page", "case_study", "testimonial",
            "faq", "product_description", "service_description", "company_bio",
            "team_bio", "mission_statement", "value_proposition", "call_to_action",
            "headline", "tagline", "slogan"
        ]
        
        # Validate configuration
        self._validate_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_env_file(self, env_file: str):
        """Load environment variables from file"""
        try:
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        except ImportError:
            logger.warning("python-dotenv not available, skipping .env file loading")
        except Exception as e:
            logger.warning(f"Failed to load {env_file}: {e}")
    
    def _validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        warnings = []
        
        # Validate AI providers
        available_providers = self.ai.get_available_providers()
        if not available_providers:
            errors.append("No AI provider API keys configured")
        
        # Validate database
        if not self.database.url or not self.database.url.startswith(("postgresql://", "sqlite://")):
            warnings.append("Database URL not properly configured")
        
        # Validate Redis
        if self.cache.strategy in [CacheStrategy.REDIS, CacheStrategy.MULTI_LEVEL]:
            if not self.redis.url or not self.redis.url.startswith("redis://"):
                warnings.append("Redis URL not properly configured")
        
        # Validate security
        if self.environment == Environment.PRODUCTION:
            if not self.security.valid_api_keys:
                warnings.append("No API keys configured for production")
            if not self.security.secret_key:
                warnings.append("No secret key configured for production")
        
        # Validate server
        if not (1024 <= self.server.port <= 65535):
            errors.append("Server port must be between 1024 and 65535")
        
        # Log validation results
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    def _setup_logging(self) -> Any:
        """Setup logging configuration"""
        log_level = getattr(logging, self.monitoring.log_level.upper(), logging.INFO)
        
        # Create formatter
        if self.monitoring.log_format == "json":
            try:
                
                class JsonFormatter(logging.Formatter):
                    def format(self, record) -> Any:
                        log_data = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "level": record.levelname,
                            "logger": record.name,
                            "message": record.getMessage(),
                            "module": record.module,
                            "function": record.funcName,
                            "line": record.lineno,
                            "environment": self.environment.value,
                            "app_name": self.app_name
                        }
                        if record.exc_info:
                            log_data["exception"] = self.formatException(record.exc_info)
                        return json.dumps(log_data)
                
                formatter = JsonFormatter()
            except ImportError:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
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
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if self.monitoring.log_file:
            try:
                file_handler = logging.FileHandler(self.monitoring.log_file)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_optimization_level(self) -> OptimizationLevel:
        """Get effective optimization level"""
        if self.optimization.level == OptimizationLevel.AUTO:
            # Auto-detect based on environment
            if self.is_production():
                return OptimizationLevel.ULTRA
            else:
                return OptimizationLevel.OPTIMIZED
        return self.optimization.level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "optimization_level": self.get_optimization_level().value,
            "cache_strategy": self.cache.strategy.value,
            "available_ai_providers": self.ai.get_available_providers(),
            "supported_languages": len(self.supported_languages),
            "supported_tones": len(self.supported_tones),
            "supported_use_cases": len(self.supported_use_cases)
        }


# Global configuration instance
_config: Optional[EnterpriseConfig] = None


def get_config(env_file: Optional[str] = None, reload: bool = False) -> EnterpriseConfig:
    """Get or create global configuration instance"""
    global _config
    
    if _config is None or reload:
        _config = EnterpriseConfig(env_file)
        logger.info(f"Configuration loaded for {_config.environment.value} environment")
    
    return _config


def reload_config(env_file: Optional[str] = None) -> EnterpriseConfig:
    """Reload configuration"""
    return get_config(env_file, reload=True) 