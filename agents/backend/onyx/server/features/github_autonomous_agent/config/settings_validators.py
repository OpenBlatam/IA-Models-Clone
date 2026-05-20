"""
Validadores para configuración de la aplicación.
"""

from pydantic import validator
from typing import List
from config.settings import Settings


class ValidatedSettings(Settings):
    """Settings con validación adicional."""
    
    @validator('PORT')
    def validate_port(cls, v):
        """Validar que el puerto esté en rango válido."""
        if not 1 <= v <= 65535:
            raise ValueError("El puerto debe estar entre 1 y 65535")
        return v
    
    @validator('WORKER_CONCURRENCY')
    def validate_worker_concurrency(cls, v):
        """Validar concurrencia del worker."""
        if v < 1:
            raise ValueError("La concurrencia del worker debe ser al menos 1")
        if v > 100:
            raise ValueError("La concurrencia del worker no puede exceder 100")
        return v
    
    @validator('TASK_POLL_INTERVAL')
    def validate_task_poll_interval(cls, v):
        """Validar intervalo de polling."""
        if v < 1:
            raise ValueError("El intervalo de polling debe ser al menos 1 segundo")
        if v > 3600:
            raise ValueError("El intervalo de polling no puede exceder 3600 segundos")
        return v
    
    @validator('CORS_ORIGINS')
    def validate_cors_origins(cls, v):
        """Validar orígenes CORS."""
        if not isinstance(v, list):
            raise ValueError("CORS_ORIGINS debe ser una lista")
        if not v:
            raise ValueError("CORS_ORIGINS no puede estar vacío")
        return v
    
    @validator('GITHUB_TOKEN')
    def validate_github_token_format(cls, v):
        """Validar formato básico del token de GitHub."""
        if v and len(v) < 10:
            raise ValueError("El token de GitHub parece inválido (muy corto)")
        return v
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v):
        """Validar que la clave secreta no sea la predeterminada en producción."""
        import os
        if os.getenv('ENVIRONMENT') == 'production' and v == 'change-me-in-production':
            raise ValueError("SECRET_KEY debe cambiarse en producción")
        return v




