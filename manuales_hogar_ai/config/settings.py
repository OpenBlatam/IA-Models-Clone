"""
Configuración del sistema.
"""

import os
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # OpenRouter
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    
    # Modelos disponibles (OpenRouter)
    default_model: str = "anthropic/claude-3.5-sonnet"
    vision_model: str = "anthropic/claude-3.5-sonnet"  # Soporta visión
    
    # Modelos alternativos
    alternative_models: List[str] = Field(
        default=[
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-70b-instruct",
            "anthropic/claude-3-opus",
        ]
    )
    
    # Configuración de generación
    max_tokens: int = Field(default=4000, ge=1, le=32000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Categorías de oficios soportadas
    supported_categories: List[str] = Field(
        default=[
            "plomeria",
            "techos",
            "carpinteria",
            "electricidad",
            "albanileria",
            "pintura",
            "herreria",
            "jardineria",
            "general"
        ]
    )
    
    @field_validator("supported_categories")
    @classmethod
    def validate_categories(cls, v: List[str]) -> List[str]:
        """Validar que las categorías sean válidas."""
        valid_categories = {
            "plomeria", "techos", "carpinteria", "electricidad",
            "albanileria", "pintura", "herreria", "jardineria", "general"
        }
        for category in v:
            if category not in valid_categories:
                raise ValueError(f"Categoría no válida: {category}")
        return v
    
    # Configuración de manuales
    manual_format: str = "lego"  # Formato tipo LEGO paso a paso
    include_images: bool = True
    include_safety_warnings: bool = True
    include_tools_list: bool = True
    include_materials_list: bool = True
    
    # Configuración de base de datos
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_name: str = os.getenv("DB_NAME", "manuales_hogar")
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    
    # Redis configuration
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "dev")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security
    allowed_origins: List[str] = Field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
    )
    api_key_header: str = Field(default="X-API-Key")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)
    rate_limit_per_hour: int = Field(default=1000, ge=1, le=100000)
    
    # Monitoring and Observability
    enable_prometheus: bool = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
    enable_tracing: bool = os.getenv("ENABLE_TRACING", "true").lower() == "true"
    otlp_endpoint: Optional[str] = os.getenv("OTLP_ENDPOINT")
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    # Circuit breaker
    circuit_breaker_failure_threshold: int = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    circuit_breaker_recovery_timeout: float = float(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60.0"))
    
    # Retry configuration
    retry_max_attempts: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))
    retry_backoff: float = float(os.getenv("RETRY_BACKOFF", "2.0"))
    
    # Serverless optimization
    cold_start_optimization: bool = os.getenv("COLD_START_OPTIMIZATION", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Obtener instancia de configuración (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

