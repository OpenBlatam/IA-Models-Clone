"""
PDF Variantes - Configuración Final del Sistema
Archivo de configuración completo para el sistema listo para usar
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """
    Configuración completa del sistema PDF Variantes
    """
    
    # Información de la aplicación
    APP_NAME: str = "PDF Variantes - Sistema Completo"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Sistema ultra-avanzado de procesamiento de PDFs con IA de próxima generación"
    DEBUG: bool = Field(False, env="DEBUG")
    ENVIRONMENT: str = Field("production", env="ENVIRONMENT")
    
    # Configuración de la API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field("super-secret-key-change-this-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Configuración de la base de datos
    DATABASE_URL: str = Field("postgresql://user:password@localhost:5432/pdf_variantes", env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(20, env="DATABASE_MAX_OVERFLOW")
    
    # Configuración de Redis
    REDIS_URL: str = Field("redis://localhost:6379", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(0, env="REDIS_DB")
    
    # Configuración de almacenamiento de archivos
    UPLOAD_PATH: str = Field("uploads", env="UPLOAD_PATH")
    VARIANTS_PATH: str = Field("variants", env="VARIANTS_PATH")
    EXPORT_PATH: str = Field("exports", env="EXPORT_PATH")
    MAX_FILE_SIZE_MB: int = Field(100, env="MAX_FILE_SIZE_MB")
    ALLOWED_FILE_TYPES: List[str] = Field(["pdf"], env="ALLOWED_FILE_TYPES")
    
    # Configuración de servicios de IA
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    
    # Modelos de IA por defecto
    DEFAULT_AI_MODEL: str = Field("gpt-3.5-turbo", env="DEFAULT_AI_MODEL")
    TOPIC_EXTRACTION_MODEL: str = Field("distilbert-base-uncased", env="TOPIC_EXTRACTION_MODEL")
    SENTIMENT_MODEL: str = Field("cardiffnlp/twitter-roberta-base-sentiment-latest", env="SENTIMENT_MODEL")
    TEXT_GENERATION_MODEL: str = Field("gpt-2", env="TEXT_GENERATION_MODEL")
    
    # Límites de procesamiento
    MAX_VARIANTS_PER_REQUEST: int = Field(1000, env="MAX_VARIANTS_PER_REQUEST")
    MAX_TOPICS_PER_DOCUMENT: int = Field(200, env="MAX_TOPICS_PER_DOCUMENT")
    MAX_BRAINSTORM_IDEAS: int = Field(500, env="MAX_BRAINSTORM_IDEAS")
    DEFAULT_VARIANT_COUNT: int = Field(10, env="DEFAULT_VARIANT_COUNT")
    
    # Configuración de caché
    CACHE_TTL_SECONDS: int = Field(3600, env="CACHE_TTL_SECONDS")
    CACHE_MAX_SIZE_MB: int = Field(1024, env="CACHE_MAX_SIZE_MB")
    ENABLE_CACHE: bool = Field(True, env="ENABLE_CACHE")
    
    # Configuración de seguridad
    CORS_ORIGINS: List[str] = Field(["*"], env="CORS_ORIGINS")
    ALLOWED_HOSTS: List[str] = Field(["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")
    RATE_LIMIT_PER_MINUTE: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(1000, env="RATE_LIMIT_PER_HOUR")
    
    # Configuración de monitoreo
    ENABLE_METRICS: bool = Field(True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(9090, env="METRICS_PORT")
    HEALTH_CHECK_INTERVAL: int = Field(30, env="HEALTH_CHECK_INTERVAL")
    
    # Configuración de logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    LOG_FILE: str = Field("logs/pdf_variantes.log", env="LOG_FILE")
    
    # Configuración de WebSocket
    WS_HEARTBEAT_INTERVAL: int = Field(30, env="WS_HEARTBEAT_INTERVAL")
    WS_MAX_CONNECTIONS: int = Field(1000, env="WS_MAX_CONNECTIONS")
    
    # Configuración de exportación
    EXPORT_FORMATS: List[str] = Field(["pdf", "docx", "txt", "html", "json", "zip", "pptx"], env="EXPORT_FORMATS")
    EXPORT_MAX_SIZE_MB: int = Field(500, env="EXPORT_MAX_SIZE_MB")
    EXPORT_TTL_HOURS: int = Field(24, env="EXPORT_TTL_HOURS")
    
    # Configuración de colaboración
    MAX_COLLABORATORS_PER_DOCUMENT: int = Field(50, env="MAX_COLLABORATORS_PER_DOCUMENT")
    COLLABORATION_SESSION_TTL_HOURS: int = Field(24, env="COLLABORATION_SESSION_TTL_HOURS")
    
    # Configuración de notificaciones
    ENABLE_NOTIFICATIONS: bool = Field(True, env="ENABLE_NOTIFICATIONS")
    NOTIFICATION_CHANNELS: List[str] = Field(["email", "push", "in_app"], env="NOTIFICATION_CHANNELS")
    
    # Configuración de analytics
    ENABLE_ANALYTICS: bool = Field(True, env="ENABLE_ANALYTICS")
    ANALYTICS_RETENTION_DAYS: int = Field(90, env="ANALYTICS_RETENTION_DAYS")
    
    # Configuración de blockchain
    BLOCKCHAIN_RPC_URL: Optional[str] = Field(None, env="BLOCKCHAIN_RPC_URL")
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = Field(None, env="BLOCKCHAIN_PRIVATE_KEY")
    IPFS_GATEWAY: str = Field("https://ipfs.io/ipfs/", env="IPFS_GATEWAY")
    
    # Configuración de plugins
    PLUGINS_DIR: str = Field("plugins", env="PLUGINS_DIR")
    ENABLE_PLUGINS: bool = Field(True, env="ENABLE_PLUGINS")
    
    # Configuración de hardware
    GPU_ENABLED: bool = Field(True, env="GPU_ENABLED")
    QUANTUM_BACKEND: str = Field("qasm_simulator", env="QUANTUM_BACKEND")
    MAX_WORKERS: int = Field(4, env="MAX_WORKERS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Configuración por defecto para desarrollo
class DevelopmentSettings(Settings):
    """Configuración para desarrollo"""
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"
    CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]

# Configuración para producción
class ProductionSettings(Settings):
    """Configuración para producción"""
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["https://yourdomain.com"]
    ALLOWED_HOSTS: List[str] = ["yourdomain.com"]

# Configuración para testing
class TestingSettings(Settings):
    """Configuración para testing"""
    DEBUG: bool = True
    ENVIRONMENT: str = "testing"
    LOG_LEVEL: str = "WARNING"
    DATABASE_URL: str = "sqlite:///test.db"
    REDIS_URL: str = "redis://localhost:6379/1"
    ENABLE_CACHE: bool = False
    ENABLE_METRICS: bool = False
    ENABLE_ANALYTICS: bool = False

def get_settings() -> Settings:
    """Obtener configuración basada en el entorno"""
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

def get_settings_by_env(env: str) -> Settings:
    """Obtener configuración por entorno específico"""
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Configuraciones específicas
class FeatureFlags:
    """Banderas de características"""
    ENABLE_ULTRA_AI: bool = True
    ENABLE_BLOCKCHAIN: bool = True
    ENABLE_PLUGINS: bool = True
    ENABLE_QUANTUM: bool = True
    ENABLE_WEB3: bool = True
    ENABLE_MONITORING: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_COLLABORATION: bool = True
    ENABLE_EXPORT: bool = True
    ENABLE_SECURITY: bool = True

class AIConfig:
    """Configuración de IA"""
    MODELS: List[str] = [
        "gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet",
        "llama-2-70b", "mistral-7b", "codellama-34b"
    ]
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TOP_K: int = 50
    REPETITION_PENALTY: float = 1.1

class SecurityConfig:
    """Configuración de seguridad"""
    MIN_PASSWORD_LENGTH: int = 8
    REQUIRE_UPPERCASE: bool = True
    REQUIRE_LOWERCASE: bool = True
    REQUIRE_NUMBERS: bool = True
    REQUIRE_SPECIAL_CHARS: bool = True
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 30
    SESSION_TIMEOUT_MINUTES: int = 60

class PerformanceConfig:
    """Configuración de rendimiento"""
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT_SECONDS: int = 30
    CONNECTION_POOL_SIZE: int = 20
    KEEP_ALIVE_TIMEOUT: int = 5
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 1

class ExportConfig:
    """Configuración de exportación"""
    SUPPORTED_FORMATS: List[str] = [
        "pdf", "docx", "txt", "html", "json", "zip", "pptx", "xlsx", "csv"
    ]
    MAX_EXPORT_SIZE_MB: int = 500
    EXPORT_TIMEOUT_SECONDS: int = 300
    COMPRESSION_LEVEL: int = 6
    INCLUDE_METADATA: bool = True
    INCLUDE_STATISTICS: bool = True

# Configuración global
settings = get_settings()
feature_flags = FeatureFlags()
ai_config = AIConfig()
security_config = SecurityConfig()
performance_config = PerformanceConfig()
export_config = ExportConfig()

# Exportar configuración
__all__ = [
    "Settings", "DevelopmentSettings", "ProductionSettings", "TestingSettings",
    "get_settings", "get_settings_by_env", "FeatureFlags", "AIConfig",
    "SecurityConfig", "PerformanceConfig", "ExportConfig",
    "settings", "feature_flags", "ai_config", "security_config",
    "performance_config", "export_config"
]