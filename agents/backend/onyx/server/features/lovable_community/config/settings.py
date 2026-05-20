"""
Main Settings class for Lovable Community

Combines all configuration sections into a single Pydantic Settings class.
"""

import os
from typing import List
from pydantic_settings import BaseSettings

from .sections import (
    AppConfig,
    DatabaseConfig,
    ServerConfig,
    CORSConfig,
    PaginationConfig,
    SearchConfig,
    ValidationConfig,
    RankingConfig,
    CacheConfig,
    SecurityConfig,
    LimitsConfig,
    NotificationsConfig,
    AnalyticsConfig,
    ExportConfig,
    TrendingConfig,
    LoggingConfig,
    AIConfig,
)


class Settings(
    AppConfig,
    DatabaseConfig,
    ServerConfig,
    CORSConfig,
    PaginationConfig,
    SearchConfig,
    ValidationConfig,
    RankingConfig,
    CacheConfig,
    SecurityConfig,
    LimitsConfig,
    NotificationsConfig,
    AnalyticsConfig,
    ExportConfig,
    TrendingConfig,
    LoggingConfig,
    AIConfig,
    BaseSettings
):
    """
    Main Settings class combining all configuration sections.
    
    Inherits from all config sections to create a unified settings object.
    """
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_settings()
    
    def _validate_settings(self) -> None:
        """
        Validates configuration after initialization.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate limits
        if self.max_page_size < 1:
            raise ValueError("max_page_size must be at least 1")
        
        if self.default_page_size > self.max_page_size:
            raise ValueError("default_page_size cannot exceed max_page_size")
        
        if self.max_chat_content_length < 1:
            raise ValueError("max_chat_content_length must be at least 1")
        
        # Validate ranking weights
        if self.vote_weight < 0 or self.remix_weight < 0 or self.view_weight < 0:
            raise ValueError("Ranking weights must be non-negative")
        
        # Validate trending periods
        valid_periods = ["hour", "day", "week", "month"]
        for period in self.trending_periods:
            if period not in valid_periods:
                raise ValueError(f"Invalid trending period: {period}")
        
        # Validate export formats
        valid_formats = ["json", "csv", "xml"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}")
        
        # Validate AI configuration
        if self.moderation_threshold < 0 or self.moderation_threshold > 1:
            raise ValueError("moderation_threshold must be between 0 and 1")
        
        if self.max_generation_length < 1:
            raise ValueError("max_generation_length must be at least 1")
        
        if self.batch_size_embeddings < 1:
            raise ValueError("batch_size_embeddings must be at least 1")


# Global settings instance
settings = Settings()








