"""
Configuración para Validación Psicológica AI
===========================================
"""

from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class PsychologicalValidationConfig(BaseSettings):
    """Configuración del sistema de validación psicológica"""
    
    # Configuración general
    app_name: str = Field(default="Psychological Validation AI", description="Nombre de la aplicación")
    app_version: str = Field(default="1.0.0", description="Versión de la aplicación")
    debug: bool = Field(default=False, description="Modo debug")
    
    # Configuración de análisis
    analysis_timeout: int = Field(default=300, description="Timeout para análisis en segundos")
    max_platforms_per_validation: int = Field(default=10, description="Máximo de plataformas por validación")
    min_confidence_score: float = Field(default=0.5, description="Score mínimo de confianza")
    
    # Configuración de caché
    cache_enabled: bool = Field(default=True, description="Habilitar caché")
    cache_ttl: int = Field(default=3600, description="TTL del caché en segundos")
    cache_max_size: int = Field(default=1000, description="Tamaño máximo del caché")
    
    # Configuración de APIs de redes sociales
    social_media_timeout: int = Field(default=30, description="Timeout para APIs de redes sociales")
    social_media_retry_attempts: int = Field(default=3, description="Intentos de reintento")
    social_media_retry_delay: int = Field(default=1, description="Delay entre reintentos en segundos")
    
    # Configuración de seguridad
    encrypt_tokens: bool = Field(default=True, description="Encriptar tokens de acceso")
    encryption_key: Optional[str] = Field(default=None, description="Clave de encriptación")
    token_expiration_buffer: int = Field(default=300, description="Buffer de expiración de tokens en segundos")
    
    # Configuración de análisis psicológico
    use_advanced_nlp: bool = Field(default=True, description="Usar NLP avanzado para análisis")
    nlp_model: str = Field(default="bert-base-multilingual-cased", description="Modelo NLP a usar")
    sentiment_analysis_enabled: bool = Field(default=True, description="Habilitar análisis de sentimientos")
    personality_analysis_enabled: bool = Field(default=True, description="Habilitar análisis de personalidad")
    
    # Configuración de base de datos
    db_pool_size: int = Field(default=10, description="Tamaño del pool de conexiones")
    db_max_overflow: int = Field(default=20, description="Máximo overflow del pool")
    db_pool_timeout: int = Field(default=30, description="Timeout del pool en segundos")
    
    # Configuración de logging
    log_level: str = Field(default="INFO", description="Nivel de logging")
    log_format: str = Field(default="json", description="Formato de logging")
    
    # Configuración de rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Habilitar rate limiting")
    rate_limit_per_minute: int = Field(default=60, description="Límite de requests por minuto")
    
    class Config:
        env_prefix = "PSYCH_VAL_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_encryption_key(self) -> str:
        """Obtener clave de encriptación"""
        if self.encryption_key:
            return self.encryption_key
        # En producción, obtener de variable de entorno segura
        return os.getenv("PSYCH_VAL_ENCRYPTION_KEY", "default-key-change-in-production")
    
    def get_social_media_config(self, platform: str) -> Dict[str, Any]:
        """Obtener configuración específica de plataforma"""
        return {
            "timeout": self.social_media_timeout,
            "retry_attempts": self.social_media_retry_attempts,
            "retry_delay": self.social_media_retry_delay,
        }


# Instancia global de configuración
config = PsychologicalValidationConfig()




