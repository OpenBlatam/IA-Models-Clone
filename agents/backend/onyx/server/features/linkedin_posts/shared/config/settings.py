from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
LinkedIn Posts Settings
======================

Configuration settings for the LinkedIn Posts system.
"""



class LinkedInPostSettings(BaseSettings):
    """
    Settings for LinkedIn Posts system.
    
    Loads configuration from environment variables with sensible defaults.
    """
    
    # API Configuration
    api_prefix: str = Field(default="/linkedin-posts", description="API prefix")
    api_version: str = Field(default="v1", description="API version")
    
    # LangChain Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    langchain_model: str = Field(default="gpt-4", description="Default LangChain model")
    langchain_temperature: float = Field(default=0.7, description="LangChain temperature")
    langchain_max_tokens: int = Field(default=2000, description="LangChain max tokens")
    
    # Content Generation
    max_content_length: int = Field(default=3000, description="Maximum content length")
    min_content_length: int = Field(default=10, description="Minimum content length")
    max_hashtags: int = Field(default=30, description="Maximum hashtags per post")
    max_keywords: int = Field(default=20, description="Maximum keywords per post")
    
    # Optimization Settings
    enable_auto_optimization: bool = Field(default=True, description="Enable auto optimization")
    optimization_timeout: int = Field(default=30, description="Optimization timeout in seconds")
    max_optimization_retries: int = Field(default=3, description="Maximum optimization retries")
    
    # Engagement Analysis
    enable_engagement_prediction: bool = Field(default=True, description="Enable engagement prediction")
    engagement_analysis_timeout: int = Field(default=15, description="Engagement analysis timeout")
    
    # A/B Testing
    max_ab_test_variants: int = Field(default=5, description="Maximum A/B test variants")
    ab_test_duration_days: int = Field(default=7, description="Default A/B test duration")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per hour")
    rate_limit_window: int = Field(default=3600, description="Rate limit window in seconds")
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")
    
    # Database
    database_url: str = Field(default="sqlite:///linkedin_posts.db", description="Database URL")
    database_pool_size: int = Field(default=10, description="Database pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow")
    
    # External Services
    linkedin_api_enabled: bool = Field(default=False, description="Enable LinkedIn API integration")
    linkedin_client_id: Optional[str] = Field(None, description="LinkedIn client ID")
    linkedin_client_secret: Optional[str] = Field(None, description="LinkedIn client secret")
    linkedin_redirect_uri: Optional[str] = Field(None, description="LinkedIn redirect URI")
    
    # Monitoring and Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_file: Optional[str] = Field(None, description="Log file path")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics port")
    
    # Security
    enable_authentication: bool = Field(default=True, description="Enable authentication")
    jwt_secret_key: str = Field(default="your-secret-key", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT expiration hours")
    
    # Content Moderation
    enable_content_moderation: bool = Field(default=True, description="Enable content moderation")
    moderation_api_key: Optional[str] = Field(None, description="Content moderation API key")
    moderation_threshold: float = Field(default=0.8, description="Moderation threshold")
    
    # Performance
    enable_async_processing: bool = Field(default=True, description="Enable async processing")
    max_concurrent_generations: int = Field(default=5, description="Max concurrent generations")
    generation_timeout: int = Field(default=60, description="Generation timeout in seconds")
    
    # Backup and Recovery
    enable_backup: bool = Field(default=True, description="Enable automatic backup")
    backup_interval_hours: int = Field(default=24, description="Backup interval in hours")
    backup_retention_days: int = Field(default=30, description="Backup retention in days")
    
    # Feature Flags
    enable_advanced_analytics: bool = Field(default=True, description="Enable advanced analytics")
    enable_social_listening: bool = Field(default=False, description="Enable social listening")
    enable_competitor_analysis: bool = Field(default=False, description="Enable competitor analysis")
    enable_ai_insights: bool = Field(default=True, description="Enable AI insights")
    
    # Content Templates
    default_templates: List[str] = Field(
        default=[
            "thought_leadership",
            "industry_insights", 
            "company_culture",
            "product_announcement",
            "educational",
            "storytelling"
        ],
        description="Default content templates"
    )
    
    # Industry-specific Settings
    industry_keywords: dict = Field(
        default={
            "technology": ["innovation", "digital transformation", "AI", "automation"],
            "finance": ["investment", "market analysis", "financial planning", "wealth management"],
            "healthcare": ["patient care", "medical innovation", "healthcare technology", "wellness"],
            "education": ["learning", "skill development", "educational technology", "professional growth"],
            "marketing": ["brand building", "customer engagement", "marketing strategy", "growth"],
            "consulting": ["business strategy", "problem solving", "client success", "consulting"]
        },
        description="Industry-specific keywords"
    )
    
    # Tone-specific Settings
    tone_guidelines: dict = Field(
        default={
            "professional": "Formal, authoritative, business-focused",
            "casual": "Relaxed, friendly, conversational",
            "friendly": "Warm, approachable, personable",
            "authoritative": "Confident, expert, leadership-focused",
            "inspirational": "Motivational, uplifting, aspirational",
            "educational": "Informative, instructional, knowledge-sharing",
            "conversational": "Engaging, interactive, discussion-oriented"
        },
        description="Tone-specific guidelines"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "LINKEDIN_POSTS_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_database_config(self) -> dict:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.log_level == "DEBUG",
        }
    
    def get_langchain_config(self) -> dict:
        """Get LangChain configuration."""
        return {
            "api_key": self.openai_api_key,
            "model": self.langchain_model,
            "temperature": self.langchain_temperature,
            "max_tokens": self.langchain_max_tokens,
        }
    
    def get_cache_config(self) -> dict:
        """Get cache configuration."""
        return {
            "enabled": self.enable_caching,
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size,
        }
    
    def get_security_config(self) -> dict:
        """Get security configuration."""
        return {
            "jwt_secret_key": self.jwt_secret_key,
            "jwt_algorithm": self.jwt_algorithm,
            "jwt_expiration_hours": self.jwt_expiration_hours,
        }
    
    def validate_settings(self) -> List[str]:
        """Validate settings and return any issues."""
        issues = []
        
        if not self.openai_api_key:
            issues.append("OpenAI API key is required")
        
        if self.max_content_length < self.min_content_length:
            issues.append("Max content length must be greater than min content length")
        
        if self.rate_limit_requests <= 0:
            issues.append("Rate limit requests must be positive")
        
        if self.cache_ttl <= 0:
            issues.append("Cache TTL must be positive")
        
        return issues 