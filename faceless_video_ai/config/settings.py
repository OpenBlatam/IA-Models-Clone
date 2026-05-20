"""
Configuration settings for Faceless Video AI with AWS support
Enhanced for cloud-native deployment
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings with AWS integration"""
    
    # API Settings
    api_title: str = "Faceless Video AI API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"  # development, staging, production
    
    # AWS Configuration
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_s3_bucket: Optional[str] = os.getenv("AWS_S3_BUCKET")
    aws_s3_prefix: str = "faceless-video"
    aws_secrets_manager_enabled: bool = os.getenv("AWS_SECRETS_MANAGER_ENABLED", "false").lower() == "true"
    aws_parameter_store_enabled: bool = os.getenv("AWS_PARAMETER_STORE_ENABLED", "false").lower() == "true"
    
    # Storage - Use S3 in production, local in development
    use_s3_storage: bool = os.getenv("USE_S3_STORAGE", "false").lower() == "true"
    output_dir: str = "/tmp/faceless_video"
    images_dir: str = "/tmp/faceless_video/images"
    audio_dir: str = "/tmp/faceless_video/audio"
    subtitles_dir: str = "/tmp/faceless_video/subtitles"
    videos_dir: str = "/tmp/faceless_video/output"
    
    # Database
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "faceless")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "changeme")
    postgres_db: str = os.getenv("POSTGRES_DB", "faceless_video")
    
    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_cache_ttl: int = 3600  # 1 hour
    
    # Celery Configuration
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    celery_worker_concurrency: int = int(os.getenv("CELERY_WORKER_CONCURRENCY", "4"))
    
    # AI Service APIs (configure with your API keys or AWS Secrets Manager)
    openai_api_key: Optional[str] = None
    stability_ai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    google_tts_enabled: bool = True
    
    # Video defaults
    default_resolution: str = "1920x1080"
    default_fps: int = 30
    default_style: str = "realistic"
    
    # Audio defaults
    default_voice: str = "neutral"
    default_speed: float = 1.0
    
    # Subtitle defaults
    default_subtitle_style: str = "modern"
    default_font_size: int = 48
    
    # Performance
    max_concurrent_generations: int = 5
    image_generation_timeout: int = 300  # seconds
    video_composition_timeout: int = 600  # seconds
    api_workers: int = int(os.getenv("API_WORKERS", "4"))
    
    # Storage
    max_video_size_mb: int = 500
    cleanup_temp_files: bool = True
    s3_presigned_url_expiry: int = 3600  # 1 hour
    
    # Security
    jwt_secret_key: Optional[str] = os.getenv("JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    cors_origins: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Monitoring & Observability
    enable_prometheus: bool = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
    enable_opentelemetry: bool = os.getenv("ENABLE_OPENTELEMETRY", "true").lower() == "true"
    opentelemetry_endpoint: Optional[str] = os.getenv("OPENTELEMETRY_ENDPOINT")
    cloudwatch_log_group: Optional[str] = os.getenv("CLOUDWATCH_LOG_GROUP")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "json"  # json or text
    enable_structured_logging: bool = True
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60  # seconds
    circuit_breaker_expected_exception: tuple = (Exception,)
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: int = 1  # seconds
    retry_backoff: float = 2.0
    
    # Serverless (Lambda) Configuration
    lambda_timeout: int = 900  # 15 minutes max for Lambda
    lambda_memory: int = 3008  # MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load secrets from AWS Secrets Manager if enabled
        if self.aws_secrets_manager_enabled:
            self._load_aws_secrets()
    
    def _load_aws_secrets(self):
        """Load secrets from AWS Secrets Manager"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            secrets_client = boto3.client('secretsmanager', region_name=self.aws_region)
            secret_name = os.getenv("AWS_SECRET_NAME", "faceless-video-ai/secrets")
            
            try:
                response = secrets_client.get_secret_value(SecretId=secret_name)
                import json
                secrets = json.loads(response['SecretString'])
                
                # Update API keys from secrets
                if not self.openai_api_key and secrets.get("OPENAI_API_KEY"):
                    self.openai_api_key = secrets["OPENAI_API_KEY"]
                if not self.stability_ai_api_key and secrets.get("STABILITY_AI_API_KEY"):
                    self.stability_ai_api_key = secrets["STABILITY_AI_API_KEY"]
                if not self.elevenlabs_api_key and secrets.get("ELEVENLABS_API_KEY"):
                    self.elevenlabs_api_key = secrets["ELEVENLABS_API_KEY"]
                if not self.jwt_secret_key and secrets.get("JWT_SECRET_KEY"):
                    self.jwt_secret_key = secrets["JWT_SECRET_KEY"]
            except ClientError as e:
                # Log error but don't fail startup
                import logging
                logging.warning(f"Failed to load AWS secrets: {e}")
        except ImportError:
            import logging
            logging.warning("boto3 not available, skipping AWS Secrets Manager")


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_database_url() -> str:
    """Get database URL from settings"""
    settings = get_settings()
    if settings.database_url:
        return settings.database_url
    return f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

