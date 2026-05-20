from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 Facebook Posts - Configuration Settings
==========================================

Configuraciones centralizadas para el sistema de Facebook posts.
"""



class Environment(str, Enum):
    """Ambientes de ejecución."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class LangChainConfig:
    """Configuración de LangChain."""
    api_key: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self) -> Any:
        if not self.api_key:
            raise ValueError("LangChain API key is required")


@dataclass
class CacheConfig:
    """Configuración de cache."""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 1000
    redis_url: str = "redis://localhost:6379"
    key_prefix: str = "facebook_posts:"


@dataclass
class AnalysisConfig:
    """Configuración de análisis."""
    confidence_threshold: float = 0.6
    quality_threshold: float = 0.5
    auto_approve_threshold: float = 0.8
    batch_size: int = 10
    parallel_processing: bool = True


@dataclass
class FacebookAPIConfig:
    """Configuración de Facebook API."""
    app_id: str
    app_secret: str
    access_token: str
    api_version: str = "v18.0"
    timeout_seconds: int = 30
    rate_limit_requests_per_hour: int = 200


@dataclass
class OnySConfig:
    """Configuración de integración con Onyx."""
    base_url: str
    api_key: str
    workspace_context_enabled: bool = True
    user_tracking_enabled: bool = True
    notification_enabled: bool = True


class FacebookPostsSettings:
    """Configuraciones principales del sistema."""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        
    """__init__ function."""
self.environment = environment
        self.debug = environment in [Environment.DEVELOPMENT, Environment.TESTING]
        
        # Load configurations
        self.langchain = self._load_langchain_config()
        self.cache = self._load_cache_config()
        self.analysis = self._load_analysis_config()
        self.facebook_api = self._load_facebook_api_config()
        self.onyx = self._load_onyx_config()
        
        # Performance settings
        self.performance = self._load_performance_config()
        
        # Feature flags
        self.features = self._load_feature_flags()
        
        # Logging configuration
        self.logging = self._load_logging_config()
    
    def _load_langchain_config(self) -> LangChainConfig:
        """Cargar configuración de LangChain."""
        return LangChainConfig(
            api_key=os.getenv("LANGCHAIN_API_KEY", ""),
            model_name=os.getenv("LANGCHAIN_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LANGCHAIN_MAX_TOKENS", "500")),
            timeout_seconds=int(os.getenv("LANGCHAIN_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("LANGCHAIN_RETRY_ATTEMPTS", "3"))
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """Cargar configuración de cache."""
        return CacheConfig(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "1000")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            key_prefix=os.getenv("CACHE_KEY_PREFIX", "facebook_posts:")
        )
    
    def _load_analysis_config(self) -> AnalysisConfig:
        """Cargar configuración de análisis."""
        return AnalysisConfig(
            confidence_threshold=float(os.getenv("ANALYSIS_CONFIDENCE_THRESHOLD", "0.6")),
            quality_threshold=float(os.getenv("ANALYSIS_QUALITY_THRESHOLD", "0.5")),
            auto_approve_threshold=float(os.getenv("ANALYSIS_AUTO_APPROVE_THRESHOLD", "0.8")),
            batch_size=int(os.getenv("ANALYSIS_BATCH_SIZE", "10")),
            parallel_processing=os.getenv("ANALYSIS_PARALLEL", "true").lower() == "true"
        )
    
    async def _load_facebook_api_config(self) -> FacebookAPIConfig:
        """Cargar configuración de Facebook API."""
        return FacebookAPIConfig(
            app_id=os.getenv("FACEBOOK_APP_ID", ""),
            app_secret=os.getenv("FACEBOOK_APP_SECRET", ""),
            access_token=os.getenv("FACEBOOK_ACCESS_TOKEN", ""),
            api_version=os.getenv("FACEBOOK_API_VERSION", "v18.0"),
            timeout_seconds=int(os.getenv("FACEBOOK_TIMEOUT", "30")),
            rate_limit_requests_per_hour=int(os.getenv("FACEBOOK_RATE_LIMIT", "200"))
        )
    
    def _load_onyx_config(self) -> OnySConfig:
        """Cargar configuración de Onyx."""
        return OnySConfig(
            base_url=os.getenv("ONYX_BASE_URL", "http://localhost:8000"),
            api_key=os.getenv("ONYX_API_KEY", ""),
            workspace_context_enabled=os.getenv("ONYX_WORKSPACE_CONTEXT", "true").lower() == "true",
            user_tracking_enabled=os.getenv("ONYX_USER_TRACKING", "true").lower() == "true",
            notification_enabled=os.getenv("ONYX_NOTIFICATIONS", "true").lower() == "true"
        )
    
    def _load_performance_config(self) -> Dict[str, Any]:
        """Cargar configuración de performance."""
        return {
            "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
            "request_timeout_seconds": int(os.getenv("REQUEST_TIMEOUT", "60")),
            "batch_processing_enabled": os.getenv("BATCH_PROCESSING", "true").lower() == "true",
            "cache_warming_enabled": os.getenv("CACHE_WARMING", "true").lower() == "true",
            "async_processing_enabled": os.getenv("ASYNC_PROCESSING", "true").lower() == "true",
            "memory_pool_size": int(os.getenv("MEMORY_POOL_SIZE", "100")),
            "connection_pool_size": int(os.getenv("CONNECTION_POOL_SIZE", "20"))
        }
    
    def _load_feature_flags(self) -> Dict[str, bool]:
        """Cargar feature flags."""
        return {
            "advanced_analytics": os.getenv("FEATURE_ADVANCED_ANALYTICS", "true").lower() == "true",
            "ab_testing": os.getenv("FEATURE_AB_TESTING", "false").lower() == "true",
            "real_time_collaboration": os.getenv("FEATURE_REAL_TIME_COLLAB", "false").lower() == "true",
            "multilingual_support": os.getenv("FEATURE_MULTILINGUAL", "false").lower() == "true",
            "auto_publishing": os.getenv("FEATURE_AUTO_PUBLISHING", "false").lower() == "true",
            "sentiment_analysis": os.getenv("FEATURE_SENTIMENT_ANALYSIS", "true").lower() == "true",
            "trend_detection": os.getenv("FEATURE_TREND_DETECTION", "true").lower() == "true",
            "competitive_analysis": os.getenv("FEATURE_COMPETITIVE_ANALYSIS", "false").lower() == "true"
        }
    
    def _load_logging_config(self) -> Dict[str, Any]:
        """Cargar configuración de logging."""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        
        return {
            "level": log_level,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_enabled": os.getenv("LOG_FILE_ENABLED", "true").lower() == "true",
            "file_path": os.getenv("LOG_FILE_PATH", "logs/facebook_posts.log"),
            "max_file_size": int(os.getenv("LOG_MAX_FILE_SIZE", "10485760")),  # 10MB
            "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
            "json_format": os.getenv("LOG_JSON_FORMAT", "false").lower() == "true",
            "include_trace": self.debug
        }
    
    def is_development(self) -> bool:
        """Verificar si está en desarrollo."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Verificar si está en producción."""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Verificar si está en testing."""
        return self.environment == Environment.TESTING
    
    def get_feature_flag(self, flag_name: str) -> bool:
        """Obtener valor de feature flag."""
        return self.features.get(flag_name, False)
    
    def validate_configuration(self) -> List[str]:
        """Validar configuración y retornar errores."""
        errors = []
        
        # Validate LangChain config
        if not self.langchain.api_key and self.environment == Environment.PRODUCTION:
            errors.append("LangChain API key is required in production")
        
        # Validate Facebook API config
        if self.get_feature_flag("auto_publishing"):
            if not self.facebook_api.app_id:
                errors.append("Facebook App ID is required for auto-publishing")
            if not self.facebook_api.app_secret:
                errors.append("Facebook App Secret is required for auto-publishing")
        
        # Validate Onyx config
        if not self.onyx.base_url:
            errors.append("Onyx base URL is required")
        
        # Validate thresholds
        if not 0 <= self.analysis.confidence_threshold <= 1:
            errors.append("Analysis confidence threshold must be between 0 and 1")
        
        if not 0 <= self.analysis.quality_threshold <= 1:
            errors.append("Analysis quality threshold must be between 0 and 1")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario (sin secrets)."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "langchain": {
                "model_name": self.langchain.model_name,
                "temperature": self.langchain.temperature,
                "max_tokens": self.langchain.max_tokens,
                "timeout_seconds": self.langchain.timeout_seconds,
                "has_api_key": bool(self.langchain.api_key)
            },
            "cache": {
                "enabled": self.cache.enabled,
                "ttl_seconds": self.cache.ttl_seconds,
                "max_entries": self.cache.max_entries,
                "key_prefix": self.cache.key_prefix
            },
            "analysis": {
                "confidence_threshold": self.analysis.confidence_threshold,
                "quality_threshold": self.analysis.quality_threshold,
                "auto_approve_threshold": self.analysis.auto_approve_threshold,
                "batch_size": self.analysis.batch_size,
                "parallel_processing": self.analysis.parallel_processing
            },
            "features": self.features,
            "performance": self.performance
        }


# Global settings instance
_settings: FacebookPostsSettings = None


def get_settings() -> FacebookPostsSettings:
    """Obtener instancia global de configuraciones."""
    global _settings
    if _settings is None:
        env = Environment(os.getenv("ENVIRONMENT", Environment.DEVELOPMENT.value))
        _settings = FacebookPostsSettings(env)
    return _settings


def reload_settings() -> FacebookPostsSettings:
    """Recargar configuraciones."""
    global _settings
    env = Environment(os.getenv("ENVIRONMENT", Environment.DEVELOPMENT.value))
    _settings = FacebookPostsSettings(env)
    return _settings


# Development/Testing helpers
def get_test_settings() -> FacebookPostsSettings:
    """Obtener configuraciones para testing."""
    return FacebookPostsSettings(Environment.TESTING)


def get_development_settings() -> FacebookPostsSettings:
    """Obtener configuraciones para desarrollo."""
    return FacebookPostsSettings(Environment.DEVELOPMENT) 