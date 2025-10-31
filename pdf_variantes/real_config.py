"""
PDF Variantes - Configuración Real Mejorada
Configuración práctica y funcional para el sistema PDF Variantes
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field

class RealSettings(BaseSettings):
    """
    Configuración real y práctica del sistema PDF Variantes
    """
    
    # Información básica de la aplicación
    APP_NAME: str = "PDF Variantes System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(False, env="DEBUG")
    ENVIRONMENT: str = Field("production", env="ENVIRONMENT")
    
    # Configuración del servidor
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    WORKERS: int = Field(4, env="WORKERS")
    
    # Configuración de la base de datos
    DATABASE_URL: str = Field("postgresql://user:password@localhost:5432/pdf_variantes", env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(20, env="DATABASE_MAX_OVERFLOW")
    DATABASE_ECHO: bool = Field(False, env="DATABASE_ECHO")
    
    # Configuración de Redis
    REDIS_URL: str = Field("redis://localhost:6379", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(0, env="REDIS_DB")
    REDIS_MAX_CONNECTIONS: int = Field(100, env="REDIS_MAX_CONNECTIONS")
    
    # Configuración de archivos
    UPLOAD_DIR: str = Field("uploads", env="UPLOAD_DIR")
    MAX_FILE_SIZE_MB: int = Field(100, env="MAX_FILE_SIZE_MB")
    ALLOWED_FILE_TYPES: List[str] = Field(["pdf"], env="ALLOWED_FILE_TYPES")
    FILE_CLEANUP_HOURS: int = Field(24, env="FILE_CLEANUP_HOURS")
    
    # Configuración de seguridad
    SECRET_KEY: str = Field("your-secret-key-change-this", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    
    # Configuración de CORS
    CORS_ORIGINS: List[str] = Field(["http://localhost:3000"], env="CORS_ORIGINS")
    CORS_ALLOW_CREDENTIALS: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    CORS_ALLOW_METHODS: List[str] = Field(["GET", "POST", "PUT", "DELETE"], env="CORS_ALLOW_METHODS")
    CORS_ALLOW_HEADERS: List[str] = Field(["*"], env="CORS_ALLOW_HEADERS")
    
    # Configuración de rate limiting
    RATE_LIMIT_ENABLED: bool = Field(True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    RATE_LIMIT_REQUESTS_PER_HOUR: int = Field(1000, env="RATE_LIMIT_REQUESTS_PER_HOUR")
    
    # Configuración de caché
    CACHE_ENABLED: bool = Field(True, env="CACHE_ENABLED")
    CACHE_TTL_SECONDS: int = Field(3600, env="CACHE_TTL_SECONDS")
    CACHE_MAX_SIZE_MB: int = Field(100, env="CACHE_MAX_SIZE_MB")
    
    # Configuración de logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    LOG_FILE: str = Field("logs/pdf_variantes.log", env="LOG_FILE")
    LOG_MAX_SIZE_MB: int = Field(10, env="LOG_MAX_SIZE_MB")
    LOG_BACKUP_COUNT: int = Field(5, env="LOG_BACKUP_COUNT")
    
    # Configuración de servicios de IA
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    
    # Configuración de modelos de IA
    DEFAULT_AI_MODEL: str = Field("gpt-3.5-turbo", env="DEFAULT_AI_MODEL")
    AI_MAX_TOKENS: int = Field(4000, env="AI_MAX_TOKENS")
    AI_TEMPERATURE: float = Field(0.7, env="AI_TEMPERATURE")
    AI_TIMEOUT_SECONDS: int = Field(30, env="AI_TIMEOUT_SECONDS")
    
    # Configuración de procesamiento
    MAX_VARIANTS_PER_REQUEST: int = Field(10, env="MAX_VARIANTS_PER_REQUEST")
    MAX_TOPICS_PER_DOCUMENT: int = Field(20, env="MAX_TOPICS_PER_DOCUMENT")
    MAX_BRAINSTORM_IDEAS: int = Field(50, env="MAX_BRAINSTORM_IDEAS")
    PROCESSING_TIMEOUT_SECONDS: int = Field(300, env="PROCESSING_TIMEOUT_SECONDS")
    
    # Configuración de exportación
    EXPORT_ENABLED: bool = Field(True, env="EXPORT_ENABLED")
    EXPORT_FORMATS: List[str] = Field(["pdf", "docx", "txt", "html", "json"], env="EXPORT_FORMATS")
    EXPORT_MAX_SIZE_MB: int = Field(500, env="EXPORT_MAX_SIZE_MB")
    EXPORT_TTL_HOURS: int = Field(24, env="EXPORT_TTL_HOURS")
    
    # Configuración de colaboración
    COLLABORATION_ENABLED: bool = Field(True, env="COLLABORATION_ENABLED")
    MAX_COLLABORATORS_PER_DOCUMENT: int = Field(10, env="MAX_COLLABORATORS_PER_DOCUMENT")
    COLLABORATION_SESSION_TTL_HOURS: int = Field(24, env="COLLABORATION_SESSION_TTL_HOURS")
    
    # Configuración de WebSocket
    WEBSOCKET_ENABLED: bool = Field(True, env="WEBSOCKET_ENABLED")
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(30, env="WEBSOCKET_HEARTBEAT_INTERVAL")
    WEBSOCKET_MAX_CONNECTIONS: int = Field(100, env="WEBSOCKET_MAX_CONNECTIONS")
    
    # Configuración de monitoreo
    MONITORING_ENABLED: bool = Field(True, env="MONITORING_ENABLED")
    METRICS_ENABLED: bool = Field(True, env="METRICS_ENABLED")
    HEALTH_CHECK_INTERVAL: int = Field(30, env="HEALTH_CHECK_INTERVAL")
    
    # Configuración de notificaciones
    NOTIFICATIONS_ENABLED: bool = Field(True, env="NOTIFICATIONS_ENABLED")
    EMAIL_ENABLED: bool = Field(False, env="EMAIL_ENABLED")
    SMTP_HOST: Optional[str] = Field(None, env="SMTP_HOST")
    SMTP_PORT: int = Field(587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(None, env="SMTP_PASSWORD")
    
    # Configuración de backup
    BACKUP_ENABLED: bool = Field(True, env="BACKUP_ENABLED")
    BACKUP_SCHEDULE: str = Field("0 2 * * *", env="BACKUP_SCHEDULE")  # Daily at 2 AM
    BACKUP_RETENTION_DAYS: int = Field(30, env="BACKUP_RETENTION_DAYS")
    BACKUP_STORAGE_PATH: str = Field("backups", env="BACKUP_STORAGE_PATH")
    
    # Configuración de SSL
    SSL_ENABLED: bool = Field(False, env="SSL_ENABLED")
    SSL_CERT_PATH: Optional[str] = Field(None, env="SSL_CERT_PATH")
    SSL_KEY_PATH: Optional[str] = Field(None, env="SSL_KEY_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Configuración por entorno
class DevelopmentSettings(RealSettings):
    """Configuración para desarrollo"""
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"
    CORS_ORIGINS: List[str] = ["*"]
    DATABASE_ECHO: bool = True
    RATE_LIMIT_ENABLED: bool = False

class ProductionSettings(RealSettings):
    """Configuración para producción"""
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["https://yourdomain.com"]
    RATE_LIMIT_ENABLED: bool = True
    SSL_ENABLED: bool = True

class TestingSettings(RealSettings):
    """Configuración para testing"""
    DEBUG: bool = True
    ENVIRONMENT: str = "testing"
    LOG_LEVEL: str = "WARNING"
    DATABASE_URL: str = "sqlite:///test.db"
    REDIS_URL: str = "redis://localhost:6379/1"
    CACHE_ENABLED: bool = False
    MONITORING_ENABLED: bool = False
    NOTIFICATIONS_ENABLED: bool = False

def get_real_settings() -> RealSettings:
    """Obtener configuración real basada en el entorno"""
    environment = os.getenv("ENVIRONMENT", "production")
    
    if environment == "development":
        return DevelopmentSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return ProductionSettings()

def validate_settings(settings: RealSettings) -> bool:
    """Validar configuración"""
    try:
        # Validar campos requeridos
        if not settings.SECRET_KEY or settings.SECRET_KEY == "your-secret-key-change-this":
            print("⚠️ SECRET_KEY debe ser configurado")
            return False
        
        if not settings.DATABASE_URL:
            print("⚠️ DATABASE_URL debe ser configurado")
            return False
        
        if not settings.REDIS_URL:
            print("⚠️ REDIS_URL debe ser configurado")
            return False
        
        # Validar directorios
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True)
        
        log_dir = Path(settings.LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)
        
        if settings.BACKUP_ENABLED:
            backup_dir = Path(settings.BACKUP_STORAGE_PATH)
            backup_dir.mkdir(exist_ok=True)
        
        print("✅ Configuración validada correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error validando configuración: {e}")
        return False

# Función para generar archivo .env
def generate_env_file(environment: str = "production", output_file: str = ".env"):
    """Generar archivo .env"""
    settings = get_real_settings()
    
    env_content = f"""# PDF Variantes - Configuración {environment.upper()}
# Generado automáticamente

# Información básica
APP_NAME="{settings.APP_NAME}"
APP_VERSION="{settings.APP_VERSION}"
DEBUG={str(settings.DEBUG).lower()}
ENVIRONMENT="{settings.ENVIRONMENT}"

# Servidor
HOST="{settings.HOST}"
PORT={settings.PORT}
WORKERS={settings.WORKERS}

# Base de datos
DATABASE_URL="{settings.DATABASE_URL}"
DATABASE_POOL_SIZE={settings.DATABASE_POOL_SIZE}
DATABASE_MAX_OVERFLOW={settings.DATABASE_MAX_OVERFLOW}
DATABASE_ECHO={str(settings.DATABASE_ECHO).lower()}

# Redis
REDIS_URL="{settings.REDIS_URL}"
REDIS_PASSWORD="{settings.REDIS_PASSWORD or ''}"
REDIS_DB={settings.REDIS_DB}
REDIS_MAX_CONNECTIONS={settings.REDIS_MAX_CONNECTIONS}

# Archivos
UPLOAD_DIR="{settings.UPLOAD_DIR}"
MAX_FILE_SIZE_MB={settings.MAX_FILE_SIZE_MB}
ALLOWED_FILE_TYPES="{','.join(settings.ALLOWED_FILE_TYPES)}"
FILE_CLEANUP_HOURS={settings.FILE_CLEANUP_HOURS}

# Seguridad
SECRET_KEY="your-secret-key-change-this"
ACCESS_TOKEN_EXPIRE_MINUTES={settings.ACCESS_TOKEN_EXPIRE_MINUTES}
REFRESH_TOKEN_EXPIRE_DAYS={settings.REFRESH_TOKEN_EXPIRE_DAYS}
ALGORITHM="{settings.ALGORITHM}"

# CORS
CORS_ORIGINS="{','.join(settings.CORS_ORIGINS)}"
CORS_ALLOW_CREDENTIALS={str(settings.CORS_ALLOW_CREDENTIALS).lower()}
CORS_ALLOW_METHODS="{','.join(settings.CORS_ALLOW_METHODS)}"
CORS_ALLOW_HEADERS="{','.join(settings.CORS_ALLOW_HEADERS)}"

# Rate Limiting
RATE_LIMIT_ENABLED={str(settings.RATE_LIMIT_ENABLED).lower()}
RATE_LIMIT_REQUESTS_PER_MINUTE={settings.RATE_LIMIT_REQUESTS_PER_MINUTE}
RATE_LIMIT_REQUESTS_PER_HOUR={settings.RATE_LIMIT_REQUESTS_PER_HOUR}

# Caché
CACHE_ENABLED={str(settings.CACHE_ENABLED).lower()}
CACHE_TTL_SECONDS={settings.CACHE_TTL_SECONDS}
CACHE_MAX_SIZE_MB={settings.CACHE_MAX_SIZE_MB}

# Logging
LOG_LEVEL="{settings.LOG_LEVEL}"
LOG_FORMAT="{settings.LOG_FORMAT}"
LOG_FILE="{settings.LOG_FILE}"
LOG_MAX_SIZE_MB={settings.LOG_MAX_SIZE_MB}
LOG_BACKUP_COUNT={settings.LOG_BACKUP_COUNT}

# IA
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
HUGGINGFACE_API_KEY=""
DEFAULT_AI_MODEL="{settings.DEFAULT_AI_MODEL}"
AI_MAX_TOKENS={settings.AI_MAX_TOKENS}
AI_TEMPERATURE={settings.AI_TEMPERATURE}
AI_TIMEOUT_SECONDS={settings.AI_TIMEOUT_SECONDS}

# Procesamiento
MAX_VARIANTS_PER_REQUEST={settings.MAX_VARIANTS_PER_REQUEST}
MAX_TOPICS_PER_DOCUMENT={settings.MAX_TOPICS_PER_DOCUMENT}
MAX_BRAINSTORM_IDEAS={settings.MAX_BRAINSTORM_IDEAS}
PROCESSING_TIMEOUT_SECONDS={settings.PROCESSING_TIMEOUT_SECONDS}

# Exportación
EXPORT_ENABLED={str(settings.EXPORT_ENABLED).lower()}
EXPORT_FORMATS="{','.join(settings.EXPORT_FORMATS)}"
EXPORT_MAX_SIZE_MB={settings.EXPORT_MAX_SIZE_MB}
EXPORT_TTL_HOURS={settings.EXPORT_TTL_HOURS}

# Colaboración
COLLABORATION_ENABLED={str(settings.COLLABORATION_ENABLED).lower()}
MAX_COLLABORATORS_PER_DOCUMENT={settings.MAX_COLLABORATORS_PER_DOCUMENT}
COLLABORATION_SESSION_TTL_HOURS={settings.COLLABORATION_SESSION_TTL_HOURS}

# WebSocket
WEBSOCKET_ENABLED={str(settings.WEBSOCKET_ENABLED).lower()}
WEBSOCKET_HEARTBEAT_INTERVAL={settings.WEBSOCKET_HEARTBEAT_INTERVAL}
WEBSOCKET_MAX_CONNECTIONS={settings.WEBSOCKET_MAX_CONNECTIONS}

# Monitoreo
MONITORING_ENABLED={str(settings.MONITORING_ENABLED).lower()}
METRICS_ENABLED={str(settings.METRICS_ENABLED).lower()}
HEALTH_CHECK_INTERVAL={settings.HEALTH_CHECK_INTERVAL}

# Notificaciones
NOTIFICATIONS_ENABLED={str(settings.NOTIFICATIONS_ENABLED).lower()}
EMAIL_ENABLED={str(settings.EMAIL_ENABLED).lower()}
SMTP_HOST=""
SMTP_PORT={settings.SMTP_PORT}
SMTP_USERNAME=""
SMTP_PASSWORD=""

# Backup
BACKUP_ENABLED={str(settings.BACKUP_ENABLED).lower()}
BACKUP_SCHEDULE="{settings.BACKUP_SCHEDULE}"
BACKUP_RETENTION_DAYS={settings.BACKUP_RETENTION_DAYS}
BACKUP_STORAGE_PATH="{settings.BACKUP_STORAGE_PATH}"

# SSL
SSL_ENABLED={str(settings.SSL_ENABLED).lower()}
SSL_CERT_PATH=""
SSL_KEY_PATH=""
"""
    
    with open(output_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Archivo .env generado: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar configuración real")
    parser.add_argument("--environment", choices=["development", "production", "testing"], 
                       default="production", help="Entorno")
    parser.add_argument("--output", default=".env", help="Archivo de salida")
    
    args = parser.parse_args()
    
    generate_env_file(args.environment, args.output)
