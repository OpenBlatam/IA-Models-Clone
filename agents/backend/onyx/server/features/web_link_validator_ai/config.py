"""Configuration for Web Link Validator AI"""
from pydantic_settings import BaseSettings
from typing import List


# Default configuration values
DEFAULT_PORT = 8025
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_CORS_ORIGINS = ["*"]


class Settings(BaseSettings):
    app_name: str = "Web Link Validator AI"
    app_version: str = "1.0.0"
    port: int = DEFAULT_PORT
    openrouter_api_key: str = ""
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    cors_origins: List[str] = DEFAULT_CORS_ORIGINS
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

