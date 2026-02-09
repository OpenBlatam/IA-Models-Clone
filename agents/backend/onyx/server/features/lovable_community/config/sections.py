"""
Configuration sections for Settings

Each section is a separate mixin class for better organization.
"""

import os
from typing import List
from pydantic import BaseModel


class AppConfig(BaseModel):
    """
    Application configuration.
    
    Attributes:
        app_name: Application name
        app_version: Application version
        debug: Debug mode flag
    """
    app_name: str = "Lovable Community API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    def model_post_init(self, __context) -> None:
        """Validate application configuration after initialization."""
        if not self.app_name or not self.app_name.strip():
            raise ValueError("app_name cannot be None or empty")
        
        if not self.app_version or not self.app_version.strip():
            raise ValueError("app_version cannot be None or empty")


class DatabaseConfig(BaseModel):
    """Database configuration"""
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./lovable_community.db"
    )
    database_echo: bool = os.getenv("DATABASE_ECHO", "False").lower() == "true"
    
    def model_post_init(self, __context) -> None:
        """Validate database configuration after initialization."""
        if not self.database_url or not self.database_url.strip():
            raise ValueError("Database URL cannot be None or empty")
        
        # Basic URL format validation
        if not any(self.database_url.strip().startswith(prefix) for prefix in ["sqlite://", "postgresql://", "mysql://", "postgres://"]):
            raise ValueError(f"Invalid database URL format: {self.database_url[:50]}...")


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8007"))
    
    def model_post_init(self, __context) -> None:
        """Validate server configuration after initialization."""
        if not self.host or not self.host.strip():
            raise ValueError("Server host cannot be None or empty")
        
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Server port must be between 1 and 65535, got {self.port}")


class CORSConfig(BaseModel):
    """CORS configuration"""
    cors_origins: List[str] = os.getenv(
        "CORS_ORIGINS",
        "*"
    ).split(",") if os.getenv("CORS_ORIGINS") != "*" else ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]


class PaginationConfig(BaseModel):
    """Pagination configuration"""
    default_page_size: int = 20
    max_page_size: int = 100
    max_page: int = 1000
    
    def model_post_init(self, __context) -> None:
        """Validate pagination configuration after initialization."""
        if self.default_page_size < 1:
            raise ValueError(f"default_page_size must be >= 1, got {self.default_page_size}")
        
        if self.max_page_size < 1:
            raise ValueError(f"max_page_size must be >= 1, got {self.max_page_size}")
        
        if self.max_page < 1:
            raise ValueError(f"max_page must be >= 1, got {self.max_page}")
        
        if self.default_page_size > self.max_page_size:
            raise ValueError(f"default_page_size ({self.default_page_size}) cannot exceed max_page_size ({self.max_page_size})")


class SearchConfig(BaseModel):
    """
    Search configuration.
    
    Attributes:
        max_search_query_length: Maximum length for search queries
        max_tags_per_chat: Maximum number of tags per chat
        max_tag_length: Maximum length for individual tags
    """
    max_search_query_length: int = 200
    max_tags_per_chat: int = 10
    max_tag_length: int = 50
    
    def model_post_init(self, __context) -> None:
        """Validate search configuration after initialization."""
        if self.max_search_query_length < 1:
            raise ValueError(f"max_search_query_length must be >= 1, got {self.max_search_query_length}")
        
        if self.max_tags_per_chat < 1:
            raise ValueError(f"max_tags_per_chat must be >= 1, got {self.max_tags_per_chat}")
        
        if self.max_tag_length < 1:
            raise ValueError(f"max_tag_length must be >= 1, got {self.max_tag_length}")


class ValidationConfig(BaseModel):
    """
    Validation configuration.
    
    Attributes:
        max_title_length: Maximum length for chat titles
        max_description_length: Maximum length for chat descriptions
        max_chat_content_length: Maximum length for chat content
    """
    max_title_length: int = 200
    max_description_length: int = 1000
    max_chat_content_length: int = 50000
    
    def model_post_init(self, __context) -> None:
        """Validate validation configuration after initialization."""
        if self.max_title_length < 1:
            raise ValueError(f"max_title_length must be >= 1, got {self.max_title_length}")
        
        if self.max_description_length < 1:
            raise ValueError(f"max_description_length must be >= 1, got {self.max_description_length}")
        
        if self.max_chat_content_length < 1:
            raise ValueError(f"max_chat_content_length must be >= 1, got {self.max_chat_content_length}")


class RankingConfig(BaseModel):
    """
    Ranking configuration.
    
    Attributes:
        vote_weight: Weight for votes in ranking calculation
        remix_weight: Weight for remixes in ranking calculation
        view_weight: Weight for views in ranking calculation
    """
    vote_weight: float = 2.0
    remix_weight: float = 3.0
    view_weight: float = 0.1
    
    def model_post_init(self, __context) -> None:
        """Validate ranking configuration after initialization."""
        if self.vote_weight < 0:
            raise ValueError(f"vote_weight must be >= 0, got {self.vote_weight}")
        
        if self.remix_weight < 0:
            raise ValueError(f"remix_weight must be >= 0, got {self.remix_weight}")
        
        if self.view_weight < 0:
            raise ValueError(f"view_weight must be >= 0, got {self.view_weight}")


class CacheConfig(BaseModel):
    """
    Cache configuration.
    
    Attributes:
        cache_enabled: Whether caching is enabled
        cache_ttl: Cache time-to-live in seconds
        cache_backend: Cache backend type (auto, redis, memory)
        redis_url: Redis URL if using Redis backend
    """
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "False").lower() == "true"
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default
    cache_backend: str = os.getenv("CACHE_BACKEND", "auto")  # auto, redis, memory
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    def model_post_init(self, __context) -> None:
        """Validate cache configuration after initialization."""
        if self.cache_ttl < 0:
            raise ValueError(f"cache_ttl must be >= 0, got {self.cache_ttl}")
        
        valid_backends = ["auto", "redis", "memory"]
        if self.cache_backend not in valid_backends:
            raise ValueError(f"cache_backend must be one of {valid_backends}, got {self.cache_backend}")
        
        if self.cache_backend == "redis" and (not self.redis_url or not self.redis_url.strip()):
            raise ValueError("redis_url cannot be empty when cache_backend is 'redis'")


class SecurityConfig(BaseModel):
    """
    Security configuration.
    
    Attributes:
        secret_key: Secret key for encryption/signing
        access_token_expire_minutes: Access token expiration time in minutes
    """
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    def model_post_init(self, __context) -> None:
        """Validate security configuration after initialization."""
        if not self.secret_key or not self.secret_key.strip():
            raise ValueError("secret_key cannot be None or empty")
        
        if self.access_token_expire_minutes < 1:
            raise ValueError(f"access_token_expire_minutes must be >= 1, got {self.access_token_expire_minutes}")


class LimitsConfig(BaseModel):
    """
    Limits configuration.
    
    Attributes:
        max_chats_per_user: Maximum chats per user
        max_remixes_per_chat: Maximum remixes per chat
        max_votes_per_user_per_day: Maximum votes per user per day
        rate_limit_enabled: Whether rate limiting is enabled
        rate_limit_requests: Number of requests allowed in rate limit window
        rate_limit_window: Rate limit window in seconds
    """
    max_chats_per_user: int = int(os.getenv("MAX_CHATS_PER_USER", "1000"))
    max_remixes_per_chat: int = int(os.getenv("MAX_REMIXES_PER_CHAT", "100"))
    max_votes_per_user_per_day: int = int(os.getenv("MAX_VOTES_PER_USER_PER_DAY", "1000"))
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "False").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    def model_post_init(self, __context) -> None:
        """Validate limits configuration after initialization."""
        if self.max_chats_per_user < 1:
            raise ValueError(f"max_chats_per_user must be >= 1, got {self.max_chats_per_user}")
        
        if self.max_remixes_per_chat < 1:
            raise ValueError(f"max_remixes_per_chat must be >= 1, got {self.max_remixes_per_chat}")
        
        if self.max_votes_per_user_per_day < 1:
            raise ValueError(f"max_votes_per_user_per_day must be >= 1, got {self.max_votes_per_user_per_day}")
        
        if self.rate_limit_requests < 1:
            raise ValueError(f"rate_limit_requests must be >= 1, got {self.rate_limit_requests}")
        
        if self.rate_limit_window < 1:
            raise ValueError(f"rate_limit_window must be >= 1, got {self.rate_limit_window}")


class NotificationsConfig(BaseModel):
    """Notifications configuration"""
    notifications_enabled: bool = os.getenv("NOTIFICATIONS_ENABLED", "True").lower() == "true"
    email_notifications: bool = os.getenv("EMAIL_NOTIFICATIONS", "False").lower() == "true"


class AnalyticsConfig(BaseModel):
    """
    Analytics configuration.
    
    Attributes:
        analytics_enabled: Whether analytics is enabled
        analytics_retention_days: Number of days to retain analytics data
    """
    analytics_enabled: bool = os.getenv("ANALYTICS_ENABLED", "True").lower() == "true"
    analytics_retention_days: int = int(os.getenv("ANALYTICS_RETENTION_DAYS", "90"))
    
    def model_post_init(self, __context) -> None:
        """Validate analytics configuration after initialization."""
        if self.analytics_retention_days < 1:
            raise ValueError(f"analytics_retention_days must be >= 1, got {self.analytics_retention_days}")


class ExportConfig(BaseModel):
    """
    Export configuration.
    
    Attributes:
        max_export_items: Maximum number of items to export
        export_formats: List of supported export formats
    """
    max_export_items: int = 100
    export_formats: List[str] = ["json", "csv", "xml"]
    
    def model_post_init(self, __context) -> None:
        """Validate export configuration after initialization."""
        if self.max_export_items < 1:
            raise ValueError(f"max_export_items must be >= 1, got {self.max_export_items}")
        
        if not self.export_formats:
            raise ValueError("export_formats cannot be empty")
        
        valid_formats = ["json", "csv", "xml"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}. Must be one of {valid_formats}")


class TrendingConfig(BaseModel):
    """
    Trending configuration.
    
    Attributes:
        trending_periods: List of valid trending periods
        trending_min_score: Minimum score for trending items
    """
    trending_periods: List[str] = ["hour", "day", "week", "month"]
    trending_min_score: float = 1.0
    
    def model_post_init(self, __context) -> None:
        """Validate trending configuration after initialization."""
        if not self.trending_periods:
            raise ValueError("trending_periods cannot be empty")
        
        valid_periods = ["hour", "day", "week", "month"]
        for period in self.trending_periods:
            if period not in valid_periods:
                raise ValueError(f"Invalid trending period: {period}. Must be one of {valid_periods}")
        
        if self.trending_min_score < 0:
            raise ValueError(f"trending_min_score must be >= 0, got {self.trending_min_score}")


class LoggingConfig(BaseModel):
    """
    Logging configuration.
    
    Attributes:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_structlog: Whether to use structured logging
        json_logs: Whether to output logs as JSON
    """
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    use_structlog: bool = os.getenv("USE_STRUCTLOG", "True").lower() == "true"
    json_logs: bool = os.getenv("JSON_LOGS", "False").lower() == "true"
    
    def model_post_init(self, __context) -> None:
        """Validate logging configuration after initialization."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {self.log_level}")


class AIConfig(BaseModel):
    """
    AI and Deep Learning configuration.
    
    Attributes:
        ai_enabled: Whether AI features are enabled
        use_gpu: Whether to use GPU
        device: Device to use (cuda or cpu)
        embedding_model: Model for embeddings
        embedding_dimension: Dimension of embeddings
        batch_size_embeddings: Batch size for embeddings
        sentiment_model: Model for sentiment analysis
        sentiment_enabled: Whether sentiment analysis is enabled
        moderation_model: Model for content moderation
        moderation_enabled: Whether moderation is enabled
        moderation_threshold: Threshold for moderation (0-1)
        text_generation_model: Model for text generation
        text_generation_enabled: Whether text generation is enabled
        max_generation_length: Maximum length for generated text
        diffusion_model: Model for diffusion
        diffusion_enabled: Whether diffusion is enabled
        use_mixed_precision: Whether to use mixed precision training
        model_cache_dir: Directory for caching models
    """
    # General AI
    ai_enabled: bool = os.getenv("AI_ENABLED", "True").lower() == "true"
    use_gpu: bool = os.getenv("USE_GPU", "True").lower() == "true"
    device: str = os.getenv("DEVICE", "cuda" if os.getenv("USE_GPU", "True").lower() == "true" else "cpu")
    
    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = 384
    batch_size_embeddings: int = int(os.getenv("BATCH_SIZE_EMBEDDINGS", "32"))
    
    # Sentiment Analysis
    sentiment_model: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiment_enabled: bool = os.getenv("SENTIMENT_ENABLED", "True").lower() == "true"
    
    # Content Moderation
    moderation_model: str = os.getenv("MODERATION_MODEL", "unitary/toxic-bert")
    moderation_enabled: bool = os.getenv("MODERATION_ENABLED", "True").lower() == "true"
    moderation_threshold: float = float(os.getenv("MODERATION_THRESHOLD", "0.7"))
    
    # Text Generation
    text_generation_model: str = os.getenv("TEXT_GENERATION_MODEL", "gpt2")
    text_generation_enabled: bool = os.getenv("TEXT_GENERATION_ENABLED", "True").lower() == "true"
    max_generation_length: int = int(os.getenv("MAX_GENERATION_LENGTH", "200"))
    
    # Diffusion Models
    diffusion_model: str = os.getenv("DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5")
    diffusion_enabled: bool = os.getenv("DIFFUSION_ENABLED", "False").lower() == "true"
    
    # Training
    use_mixed_precision: bool = os.getenv("USE_MIXED_PRECISION", "True").lower() == "true"
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models_cache")
    
    def model_post_init(self, __context) -> None:
        """Validate AI configuration after initialization."""
        valid_devices = ["cuda", "cpu"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")
        
        if not self.embedding_model or not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be None or empty")
        
        if self.embedding_dimension < 1:
            raise ValueError(f"embedding_dimension must be >= 1, got {self.embedding_dimension}")
        
        if self.batch_size_embeddings < 1:
            raise ValueError(f"batch_size_embeddings must be >= 1, got {self.batch_size_embeddings}")
        
        if not self.sentiment_model or not self.sentiment_model.strip():
            raise ValueError("sentiment_model cannot be None or empty")
        
        if not self.moderation_model or not self.moderation_model.strip():
            raise ValueError("moderation_model cannot be None or empty")
        
        if not (0.0 <= self.moderation_threshold <= 1.0):
            raise ValueError(f"moderation_threshold must be between 0.0 and 1.0, got {self.moderation_threshold}")
        
        if not self.text_generation_model or not self.text_generation_model.strip():
            raise ValueError("text_generation_model cannot be None or empty")
        
        if self.max_generation_length < 1:
            raise ValueError(f"max_generation_length must be >= 1, got {self.max_generation_length}")
        
        if not self.diffusion_model or not self.diffusion_model.strip():
            raise ValueError("diffusion_model cannot be None or empty")
        
        if not self.model_cache_dir or not self.model_cache_dir.strip():
            raise ValueError("model_cache_dir cannot be None or empty")








