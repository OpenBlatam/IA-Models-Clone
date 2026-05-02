"""
Configuración de la aplicación.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List
import os
import secrets


class Settings(BaseSettings):
    """Configuración de la aplicación con validación mejorada."""

    # Server
    HOST: str = Field(default="0.0.0.0", description="Host del servidor")
    PORT: int = Field(default=8030, ge=1, le=65535, description="Puerto del servidor")
    DEBUG: bool = Field(default=False, description="Modo debug")

    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
        ],
        description="Orígenes permitidos para CORS"
    )

    # GitHub
    GITHUB_TOKEN: str = Field(default="", description="Token de autenticación de GitHub")
    GITHUB_API_BASE_URL: str = Field(
        default="https://api.github.com",
        description="URL base de la API de GitHub"
    )
    GITHUB_API_TIMEOUT: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout en segundos para llamadas a la API de GitHub"
    )

    # Redis (para Celery)
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="URL de conexión a Redis"
    )
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="URL del broker de Celery"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        description="URL del backend de resultados de Celery"
    )

    # Database
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./github_agent.db",
        description="URL de conexión a la base de datos"
    )
    DATABASE_TIMEOUT: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout en segundos para operaciones de base de datos"
    )

    # Worker
    WORKER_CONCURRENCY: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Número de workers concurrentes"
    )
    WORKER_MAX_TASKS_PER_CHILD: int = Field(
        default=1000,
        ge=1,
        description="Máximo de tareas por worker antes de reiniciar"
    )
    TASK_POLL_INTERVAL: int = Field(
        default=5,
        ge=1,
        le=300,
        description="Intervalo en segundos para verificar nuevas tareas"
    )
    TASK_TIMEOUT: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Timeout en segundos para ejecutar una tarea"
    )

    # Storage
    STORAGE_PATH: str = Field(
        default="./storage",
        description="Ruta base para almacenamiento"
    )
    TASKS_STORAGE_PATH: str = Field(
        default="./storage/tasks",
        description="Ruta para almacenar tareas"
    )
    LOGS_STORAGE_PATH: str = Field(
        default="./storage/logs",
        description="Ruta para almacenar logs"
    )

    # Security
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32) if os.getenv("SECRET_KEY") is None else os.getenv("SECRET_KEY"),
        description="Clave secreta para encriptación (generada automáticamente si no se proporciona)"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Tiempo de expiración de tokens de acceso en minutos"
    )

    # Circuit Breaker
    CIRCUIT_BREAKER_MAX_FAILURES: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número máximo de fallos antes de abrir el circuit breaker"
    )
    CIRCUIT_BREAKER_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Tiempo en segundos antes de intentar recuperar el circuit breaker"
    )

    # OpenRouter / LLM Configuration
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="API key de OpenRouter para acceso a modelos de IA"
    )
    DEEPSEEK_API_KEY: str = Field(
        default="",
        description="API key nativa de DeepSeek"
    )
    DEEPSEEK_API_BASE_URL: str = Field(
        default="https://api.deepseek.com",
        description="URL base para la API de DeepSeek"
    )
    OPENROUTER_HTTP_REFERER: str = Field(
        default="",
        description="HTTP Referer para requests a OpenRouter (opcional)"
    )
    OPENROUTER_X_TITLE: str = Field(
        default="GitHub Autonomous Agent",
        description="Título de la aplicación para OpenRouter"
    )
    LLM_DEFAULT_MODELS: List[str] = Field(
        default=[
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5"
        ],
        description="Lista de modelos LLM por defecto para ejecución paralela"
    )
    LLM_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout en segundos para requests a modelos LLM"
    )
    LLM_MAX_PARALLEL_REQUESTS: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Máximo número de requests paralelos a modelos LLM"
    )
    LLM_ENABLED: bool = Field(
        default=True,
        description="Habilitar uso de modelos LLM (requiere OPENROUTER_API_KEY)"
    )

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validar que la clave secreta tenga suficiente longitud."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY debe tener al menos 32 caracteres")
        return v

    @field_validator("GITHUB_TOKEN")
    @classmethod
    def validate_github_token(cls, v: str) -> str:
        """Validar formato básico del token de GitHub."""
        if v and not v.startswith(("ghp_", "gho_", "ghu_", "ghs_", "ghr_")):
            # Permitir tokens personalizados pero advertir
            import warnings
            warnings.warn("El token de GitHub no parece tener el formato estándar")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"


settings = Settings()


