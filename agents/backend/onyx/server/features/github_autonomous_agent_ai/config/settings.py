"""
Configuración del sistema.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8025"))
    API_TITLE: str = os.getenv("API_TITLE", "GitHub Autonomous Agent AI")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
    ).split(",") if isinstance(os.getenv("ALLOWED_ORIGINS", ""), str) else [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000"
    ]
    
    # GitHub Settings
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
    GITHUB_CLIENT_ID: Optional[str] = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET: Optional[str] = os.getenv("GITHUB_CLIENT_SECRET")
    GITHUB_REDIRECT_URI: str = os.getenv(
        "GITHUB_REDIRECT_URI",
        f"http://localhost:{int(os.getenv('API_PORT', '8025'))}/api/github/auth/callback"
    )
    GITHUB_API_BASE_URL: str = os.getenv("GITHUB_API_BASE_URL", "https://api.github.com")
    
    # Agent Settings
    AGENT_POLL_INTERVAL: int = int(os.getenv("AGENT_POLL_INTERVAL", "5"))
    AGENT_MAX_CONCURRENT_TASKS: int = int(os.getenv("AGENT_MAX_CONCURRENT_TASKS", "3"))
    AGENT_TASK_TIMEOUT: int = int(os.getenv("AGENT_TASK_TIMEOUT", "3600"))
    
    # Storage Settings
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./data")
    TASKS_DB_PATH: str = os.getenv("TASKS_DB_PATH", "./data/tasks.db")
    
    # Service Settings
    SERVICE_ENABLED: bool = os.getenv("SERVICE_ENABLED", "true").lower() == "true"
    SERVICE_AUTO_RESTART: bool = os.getenv("SERVICE_AUTO_RESTART", "true").lower() == "true"
    
    # LLM Settings (DeepSeek)
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE_URL: str = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").lower() == "true"
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
