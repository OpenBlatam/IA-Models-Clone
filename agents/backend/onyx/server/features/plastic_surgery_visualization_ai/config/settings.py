"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8025
    log_level: str = "INFO"
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    
    # AI Model settings
    model_provider: str = "openai"  # openai, anthropic, local
    model_name: str = "gpt-4-vision-preview"
    api_key: str = ""
    
    # Image processing settings
    max_image_size_mb: int = 10
    supported_formats: List[str] = ["jpg", "jpeg", "png", "webp"]
    output_format: str = "png"
    
    # Storage settings
    upload_dir: str = "./storage/uploads"
    output_dir: str = "./storage/outputs"
    
    # Processing settings
    max_concurrent_requests: int = 5
    request_timeout: int = 300  # seconds
    batch_max_concurrent: int = 3
    
    # Retry settings
    retry_max_attempts: int = 3
    retry_initial_wait: float = 1.0
    retry_max_wait: float = 10.0
    
    # Performance settings
    enable_performance_monitoring: bool = True
    log_slow_requests: bool = True
    slow_request_threshold: float = 5.0  # seconds
    
    # Logging settings
    use_json_logging: bool = False
    log_requests: bool = True
    
    # Environment
    environment: str = "development"  # development, production, testing
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

