"""Settings configuration for Logistics AI Platform"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8030
    LOG_LEVEL: str = "INFO"
    
    # Database
    DATABASE_URL: Optional[str] = "sqlite+aiosqlite:///./logistics.db"
    
    # Redis (for caching and real-time tracking)
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"
    
    # External APIs
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    WEATHER_API_KEY: Optional[str] = None
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # Tracking
    TRACKING_UPDATE_INTERVAL: int = 60  # seconds
    GPS_UPDATE_INTERVAL: int = 30  # seconds
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".jpg", ".jpeg", ".png", ".doc", ".docx", ".xls", ".xlsx"]
    
    # Email (for notifications)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()













