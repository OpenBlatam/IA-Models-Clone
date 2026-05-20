"""
Real AI Document Processor Configuration
Practical configuration for the document processing system
"""

from pydantic import BaseSettings
from typing import Optional
import os

class RealAISettings(BaseSettings):
    """Real AI Document Processor Settings"""
    
    # Application settings
    app_name: str = "Real AI Document Processor"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # AI Model settings
    spacy_model: str = "en_core_web_sm"
    transformers_cache_dir: Optional[str] = None
    max_text_length: int = 5120
    max_summary_length: int = 150
    min_summary_length: int = 30
    
    # Processing settings
    enable_spacy: bool = True
    enable_nltk: bool = True
    enable_transformers: bool = True
    enable_sentiment: bool = True
    enable_classification: bool = True
    enable_summarization: bool = True
    enable_qa: bool = True
    
    # API settings
    rate_limit_per_minute: int = 100
    max_file_size_mb: int = 10
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database settings (optional)
    database_url: Optional[str] = None
    
    # Redis settings (optional)
    redis_url: Optional[str] = None
    
    # OpenAI settings (optional)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Cloud storage settings (optional)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    aws_s3_bucket: Optional[str] = None
    
    google_cloud_project: Optional[str] = None
    google_cloud_bucket: Optional[str] = None
    
    azure_storage_account: Optional[str] = None
    azure_storage_key: Optional[str] = None
    azure_container: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = RealAISettings()













