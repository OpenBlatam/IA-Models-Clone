"""
PDF Variantes Configuration
Settings and configuration management
"""

import os
from typing import Optional, List, Dict, Any
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "PDF Variantes API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./pdf_variantes.db", env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # File Storage
    UPLOAD_PATH: str = "uploads"
    VARIANTS_PATH: str = "variants"
    EXPORT_PATH: str = "exports"
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = ["pdf"]
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # AI Models
    DEFAULT_AI_MODEL: str = "gpt-3.5-turbo"
    TOPIC_EXTRACTION_MODEL: str = "distilbert-base-uncased"
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    TEXT_GENERATION_MODEL: str = "gpt-2"
    
    # Processing
    MAX_VARIANTS_PER_REQUEST: int = 1000
    MAX_TOPICS_PER_DOCUMENT: int = 200
    MAX_BRAINSTORM_IDEAS: int = 500
    DEFAULT_VARIANT_COUNT: int = 10
    
    # Caching
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_SIZE_MB: int = 1024
    ENABLE_CACHE: bool = True
    
    # Security
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100  # For main.py compatibility
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 1000
    
    # Export
    EXPORT_FORMATS: List[str] = ["pdf", "docx", "txt", "html", "json", "zip", "pptx"]
    EXPORT_MAX_SIZE_MB: int = 500
    EXPORT_TTL_HOURS: int = 24
    
    # Collaboration
    MAX_COLLABORATORS_PER_DOCUMENT: int = 50
    COLLABORATION_SESSION_TTL_HOURS: int = 24
    
    # Notifications
    ENABLE_NOTIFICATIONS: bool = True
    NOTIFICATION_CHANNELS: List[str] = ["email", "push", "in_app"]
    
    # Analytics
    ENABLE_ANALYTICS: bool = True
    ANALYTICS_RETENTION_DAYS: int = 90
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """Get application settings"""
    return Settings()

# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"

class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "WARNING"
    ENABLE_METRICS: bool = True

class TestingSettings(Settings):
    """Testing environment settings"""
    DEBUG: bool = True
    ENVIRONMENT: str = "testing"
    DATABASE_URL: str = "sqlite:///./test.db"
    REDIS_URL: str = "redis://localhost:6379/1"
    ENABLE_CACHE: bool = False
    ENABLE_METRICS: bool = False

def get_settings_by_env(env: str = None) -> Settings:
    """Get settings based on environment"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Feature flags
class FeatureFlags:
    """Feature flags for the application"""
    
    ENABLE_AI_ENHANCEMENTS: bool = True
    ENABLE_REAL_TIME_COLLABORATION: bool = True
    ENABLE_ADVANCED_EXPORT: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_NOTIFICATIONS: bool = True
    ENABLE_WEBHOOKS: bool = True
    ENABLE_BATCH_PROCESSING: bool = True
    ENABLE_SEARCH: bool = True
    ENABLE_CACHING: bool = True
    ENABLE_SECURITY_SCANNING: bool = True
    
    @classmethod
    def get_all_flags(cls) -> Dict[str, bool]:
        """Get all feature flags"""
        return {
            "ai_enhancements": cls.ENABLE_AI_ENHANCEMENTS,
            "real_time_collaboration": cls.ENABLE_REAL_TIME_COLLABORATION,
            "advanced_export": cls.ENABLE_ADVANCED_EXPORT,
            "analytics": cls.ENABLE_ANALYTICS,
            "notifications": cls.ENABLE_NOTIFICATIONS,
            "webhooks": cls.ENABLE_WEBHOOKS,
            "batch_processing": cls.ENABLE_BATCH_PROCESSING,
            "search": cls.ENABLE_SEARCH,
            "caching": cls.ENABLE_CACHING,
            "security_scanning": cls.ENABLE_SECURITY_SCANNING
        }

# AI Configuration
class AIConfig:
    """AI service configuration"""
    
    # OpenAI
    OPENAI_MODELS = {
        "gpt-3.5-turbo": {"max_tokens": 4096, "cost_per_1k": 0.002},
        "gpt-4": {"max_tokens": 8192, "cost_per_1k": 0.03},
        "gpt-4-turbo": {"max_tokens": 128000, "cost_per_1k": 0.01}
    }
    
    # Anthropic
    ANTHROPIC_MODELS = {
        "claude-3-sonnet": {"max_tokens": 200000, "cost_per_1k": 0.003},
        "claude-3-opus": {"max_tokens": 200000, "cost_per_1k": 0.015}
    }
    
    # Hugging Face
    HUGGINGFACE_MODELS = {
        "topic_extraction": "distilbert-base-uncased",
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "text_generation": "gpt2",
        "summarization": "facebook/bart-large-cnn"
    }
    
    @classmethod
    def get_model_config(cls, provider: str, model: str) -> Dict[str, Any]:
        """Get model configuration"""
        if provider == "openai":
            return cls.OPENAI_MODELS.get(model, {})
        elif provider == "anthropic":
            return cls.ANTHROPIC_MODELS.get(model, {})
        elif provider == "huggingface":
            return cls.HUGGINGFACE_MODELS.get(model, {})
        return {}

# Security Configuration
class SecurityConfig:
    """Security configuration"""
    
    # Password requirements
    MIN_PASSWORD_LENGTH: int = 8
    REQUIRE_UPPERCASE: bool = True
    REQUIRE_LOWERCASE: bool = True
    REQUIRE_NUMBERS: bool = True
    REQUIRE_SPECIAL_CHARS: bool = True
    
    # Session security
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15
    
    # File security
    SCAN_UPLOADED_FILES: bool = True
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_MIME_TYPES: List[str] = ["application/pdf"]
    
    # Rate limiting
    RATE_LIMIT_STRATEGIES = {
        "fixed_window": {"window_size": 60, "max_requests": 60},
        "sliding_window": {"window_size": 60, "max_requests": 60},
        "token_bucket": {"capacity": 60, "refill_rate": 1},
        "leaky_bucket": {"capacity": 60, "leak_rate": 1}
    }

# Performance Configuration
class PerformanceConfig:
    """Performance configuration"""
    
    # Caching
    CACHE_STRATEGIES = {
        "lru": {"max_size": 1000},
        "lfu": {"max_size": 1000},
        "fifo": {"max_size": 1000},
        "ttl": {"default_ttl": 3600}
    }
    
    # Database
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    
    # Processing
    MAX_CONCURRENT_UPLOADS: int = 10
    MAX_CONCURRENT_PROCESSING: int = 5
    PROCESSING_TIMEOUT_SECONDS: int = 300
    
    # Memory
    MAX_MEMORY_USAGE_MB: int = 2048
    MEMORY_CLEANUP_INTERVAL: int = 300

# Export configuration
class ExportConfig:
    """Export configuration"""
    
    SUPPORTED_FORMATS = {
        "pdf": {
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "max_size_mb": 100
        },
        "docx": {
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "extension": ".docx",
            "max_size_mb": 50
        },
        "txt": {
            "mime_type": "text/plain",
            "extension": ".txt",
            "max_size_mb": 10
        },
        "html": {
            "mime_type": "text/html",
            "extension": ".html",
            "max_size_mb": 10
        },
        "json": {
            "mime_type": "application/json",
            "extension": ".json",
            "max_size_mb": 10
        },
        "zip": {
            "mime_type": "application/zip",
            "extension": ".zip",
            "max_size_mb": 500
        },
        "pptx": {
            "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "extension": ".pptx",
            "max_size_mb": 100
        }
    }
    
    @classmethod
    def get_format_config(cls, format_name: str) -> Dict[str, Any]:
        """Get format configuration"""
        return cls.SUPPORTED_FORMATS.get(format_name, {})

# Default configurations
DEFAULT_CONFIG = {
    "ai": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "max_tokens": 2000,
        "temperature": 0.7
    },
    "processing": {
        "max_variants": 10,
        "max_topics": 50,
        "max_brainstorm_ideas": 20,
        "timeout_seconds": 300
    },
    "export": {
        "default_format": "pdf",
        "include_metadata": True,
        "include_statistics": True,
        "compress": False
    },
    "collaboration": {
        "max_users": 50,
        "session_timeout": 24,
        "enable_chat": True,
        "enable_annotations": True
    }
}

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return DEFAULT_CONFIG.copy()
