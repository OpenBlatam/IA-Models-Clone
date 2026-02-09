"""
Settings and configuration

This module defines the application settings using Pydantic,
with support for environment variables and validation.

All settings can be configured via environment variables with the PSD_ prefix.
For example: PSD_OPENAI_API_KEY, PSD_HOST, PSD_PORT, etc.

Example:
    ```python
    from ..config.settings import settings
    
    # Access settings
    api_key = settings.openai_api_key
    host = settings.host
    port = settings.port
    ```
"""

from pathlib import Path
from typing import Optional, List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key para generación con LLM"
    )
    
    # Server
    host: str = Field(
        default="0.0.0.0",
        description="Host del servidor"
    )
    port: int = Field(
        default=8030,
        ge=1,
        le=65535,
        description="Puerto del servidor"
    )
    reload: bool = Field(
        default=False,
        description="Habilitar auto-reload en desarrollo"
    )
    
    # CORS
    cors_origins: Union[str, List[str]] = Field(
        default=["*"],
        description="Orígenes permitidos para CORS"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Permitir credenciales en CORS"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Nivel de logging"
    )
    log_format: str = Field(
        default="json",
        description="Formato de logs: 'json' o 'text'"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Archivo para logs (opcional)"
    )
    
    # Security
    api_key_header: Optional[str] = Field(
        default=None,
        description="Header para API key (opcional)"
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Límite de requests por minuto"
    )
    secret_key: Optional[str] = Field(
        default=None,
        description="Secret key para JWT (generar en producción)"
    )
    
    # Storage
    storage_path: str = Field(
        default="storage",
        description="Ruta base para almacenamiento"
    )
    designs_path: str = Field(
        default="storage/designs",
        description="Ruta para almacenar diseños"
    )
    
    # Performance
    max_workers: int = Field(
        default=4,
        ge=1,
        description="Número máximo de workers"
    )
    request_timeout: int = Field(
        default=300,
        ge=1,
        description="Timeout de requests en segundos"
    )
    
    # Feature flags
    enable_ml_features: bool = Field(
        default=False,
        description="Habilitar características ML avanzadas"
    )
    enable_deep_learning: bool = Field(
        default=False,
        description="Habilitar deep learning"
    )
    enable_health_checks: bool = Field(
        default=True,
        description="Habilitar health checks"
    )
    
    # Environment
    environment: str = Field(
        default="development",
        description="Entorno: 'development', 'staging', 'production'"
    )
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """
        Parse CORS origins from string or list.
        
        Args:
            v: String with comma-separated origins or list of origins
            
        Returns:
            List of origin strings
        """
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return [str(origin).strip() for origin in v if origin]
        return []
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """
        Validate log level.
        
        Args:
            v: Log level string
            
        Returns:
            Uppercase log level string
            
        Raises:
            ValueError: If log level is invalid
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level debe ser uno de: {valid_levels}")
        return v_upper
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """
        Validate log format.
        
        Args:
            v: Log format string ('json' or 'text')
            
        Returns:
            Lowercase log format string
            
        Raises:
            ValueError: If log format is invalid
        """
        valid_formats = ["json", "text"]
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"log_format debe ser uno de: {valid_formats}")
        return v_lower
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """
        Validate environment.
        
        Args:
            v: Environment string
            
        Returns:
            Lowercase environment string
            
        Raises:
            ValueError: If environment is invalid
        """
        valid_envs = ["development", "staging", "production"]
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"environment debe ser uno de: {valid_envs}")
        return v_lower
    
    @property
    def is_production(self) -> bool:
        """Verificar si está en producción"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Verificar si está en desarrollo"""
        return self.environment == "development"
    
    def ensure_directories(self) -> None:
        """
        Create necessary directories if they don't exist.
        
        This method ensures that all storage paths and log file directories
        are created before the application starts using them.
        """
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.designs_path).mkdir(parents=True, exist_ok=True)
        if self.log_file:
            log_path = Path(self.log_file)
            if log_path.parent != Path("."):
                log_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="PSD_",
        extra="ignore",  # Ignore extra fields from environment
    )


settings = Settings()
settings.ensure_directories()

