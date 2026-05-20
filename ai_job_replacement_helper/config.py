"""
Configuration module
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # LinkedIn API
    linkedin_api_key: Optional[str] = os.getenv("LINKEDIN_API_KEY")
    linkedin_api_secret: Optional[str] = os.getenv("LINKEDIN_API_SECRET")
    
    # Database
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    
    # Redis
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # App
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8030"))
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "change-this-secret-key")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()




