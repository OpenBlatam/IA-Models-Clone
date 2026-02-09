"""
Enhanced Configuration for Advanced Content Redundancy Detector
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Enhanced configuration for the application"""
    
    # Application Configuration
    app_name: str = "Advanced Content Redundancy Detector"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Analysis Configuration
    default_similarity_threshold: float = 0.8
    max_content_length: int = 50000
    min_content_length: int = 10
    max_batch_size: int = 100
    
    # AI/ML Configuration
    model_cache_size: int = 10
    enable_gpu: bool = False
    model_timeout: int = 30
    embedding_model: str = "all-MiniLM-L6-v2"
    language_model: str = "distilbert-base-uncased"
    
    # Database Configuration
    database_url: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    
    # Security Configuration
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    cors_origins: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    rate_limit_burst: int = 20
    
    # Monitoring Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_metrics: bool = True
    metrics_port: int = 9090
    sentry_dsn: Optional[str] = None
    
    # Performance Configuration
    request_timeout: int = 30
    max_concurrent_requests: int = 100
    enable_compression: bool = True
    compression_level: int = 6
    
    # Feature Flags
    enable_sentiment_analysis: bool = True
    enable_topic_modeling: bool = True
    enable_semantic_analysis: bool = True
    enable_language_detection: bool = True
    enable_plagiarism_detection: bool = True
    enable_quality_scoring: bool = True
    
    # Batch Processing
    enable_batch_processing: bool = True
    batch_queue_size: int = 1000
    batch_processing_interval: int = 5
    
    # Export Configuration
    enable_export: bool = True
    export_formats: List[str] = ["json", "csv", "xlsx", "pdf"]
    max_export_size: int = 10000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
settings = Settings()



