"""
Configuración del proyecto
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_http_referer: str = "https://blatam-academy.com"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Cache
    cache_max_size: int = 1000
    cache_ttl: int = 3600
    
    # Scraper
    scraper_timeout: float = 30.0
    scraper_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Logging
    log_level: str = "INFO"
    
    # Rate Limiting
    rate_limit_max_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()

