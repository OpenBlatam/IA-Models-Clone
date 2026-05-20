"""
Configuration for Unified AI Model
Combines best practices from all agent implementations
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers via OpenRouter."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    META = "meta"
    MISTRAL = "mistral"
    COHERE = "cohere"


@dataclass
class DeepSeekConfig:
    """DeepSeek API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    api_base: str = "https://api.deepseek.com/v1"
    timeout: float = 60.0
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0


@dataclass
class OpenRouterConfig:
    """OpenRouter API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    api_base: str = "https://openrouter.ai/api/v1"
    http_referer: str = field(default_factory=lambda: os.getenv("OPENROUTER_HTTP_REFERER", "https://blatam-academy.com"))
    app_name: str = field(default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "Unified AI Model"))
    timeout: float = 60.0
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    ttl: int = 3600  # 1 hour
    max_size: int = 10000
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    window_seconds: int = 60


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 10  # More lenient
    recovery_timeout: float = 5.0  # Faster recovery
    success_threshold: int = 1  # Only need 1 success to close


@dataclass
class ChatConfig:
    """Chat configuration."""
    max_history_length: int = 50
    max_context_tokens: int = 8000
    default_system_prompt: str = "You are a helpful AI assistant."
    conversation_ttl: int = 3600  # 1 hour


@dataclass
class AutonomousConfig:
    """Configuration for autonomous agent features."""
    learning_enabled: bool = True
    learning_adaptation_rate: float = 0.1
    enable_world_model: bool = True
    enable_self_reflection: bool = True
    enable_experience_learning: bool = True
    max_knowledge_entries: int = 10000
    knowledge_base_retention_days: int = 30
    agent_poll_interval: float = 1.0



@dataclass
class UnifiedAIConfig:
    """
    Main configuration for Unified AI Model.
    Combines all sub-configurations.
    """
    # Server settings
    host: str = field(default_factory=lambda: os.getenv("UNIFIED_AI_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("UNIFIED_AI_PORT", "8050")))
    debug: bool = field(default_factory=lambda: os.getenv("UNIFIED_AI_DEBUG", "false").lower() == "true")
    
    # Default models
    default_model: str = field(default_factory=lambda: os.getenv(
        "UNIFIED_AI_DEFAULT_MODEL", "deepseek-chat"
    ))
    default_models: List[str] = field(default_factory=lambda: [
        "deepseek-chat",
        "deepseek-coder"
    ])
    
    # Model parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 4000
    max_parallel_requests: int = 10
    
    # Sub-configurations
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    autonomous: AutonomousConfig = field(default_factory=AutonomousConfig)
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090
    
    @property
    def use_openrouter(self) -> bool:
        """Check if OpenRouter API should be used (preferred if available)."""
        return bool(self.openrouter.api_key)
    
    @property
    def use_deepseek(self) -> bool:
        """Check if DeepSeek API should be used (only if OpenRouter not available)."""
        # Prefer OpenRouter if available, fallback to DeepSeek
        if self.openrouter.api_key:
            return False
        return bool(self.deepseek.api_key)
    
    @property
    def active_api_key(self) -> str:
        """Get the active API key (OpenRouter preferred, DeepSeek fallback)."""
        if self.openrouter.api_key:
            return self.openrouter.api_key
        return self.deepseek.api_key
    
    @property
    def active_api_base(self) -> str:
        """Get the active API base URL."""
        if self.openrouter.api_key:
            return self.openrouter.api_base
        return self.deepseek.api_base
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.deepseek.api_key and not self.openrouter.api_key:
            import warnings
            warnings.warn("No API key set (DEEPSEEK_API_KEY or OPENROUTER_API_KEY). LLM functionality will be limited.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "default_model": self.default_model,
            "default_models": self.default_models,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "max_parallel_requests": self.max_parallel_requests,
            "cache_enabled": self.cache.enabled,
            "rate_limit_enabled": self.rate_limit.enabled,
            "metrics_enabled": self.metrics_enabled,
        }
    
    @classmethod
    def from_env(cls) -> "UnifiedAIConfig":
        """Create configuration from environment variables."""
        return cls()


# Global configuration instance
_config: Optional[UnifiedAIConfig] = None


def get_config() -> UnifiedAIConfig:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = UnifiedAIConfig.from_env()
    return _config


def set_config(config: UnifiedAIConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config



