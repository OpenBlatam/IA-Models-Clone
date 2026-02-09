"""
Excepciones personalizadas para Validación Psicológica AI
==========================================================
"""

from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger()


class PsychologicalValidationError(Exception):
    """Excepción base para errores de validación psicológica"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
        logger.error(
            "PsychologicalValidationError",
            message=message,
            error_code=error_code,
            details=details
        )


class SocialMediaConnectionError(PsychologicalValidationError):
    """Error al conectar con red social"""
    pass


class SocialMediaAPIError(PsychologicalValidationError):
    """Error al llamar a API de red social"""
    pass


class ValidationNotFoundError(PsychologicalValidationError):
    """Validación no encontrada"""
    pass


class ValidationAlreadyRunningError(PsychologicalValidationError):
    """Validación ya está en ejecución"""
    pass


class InsufficientDataError(PsychologicalValidationError):
    """Datos insuficientes para análisis"""
    pass


class AnalysisTimeoutError(PsychologicalValidationError):
    """Timeout en análisis"""
    pass


class ProfileGenerationError(PsychologicalValidationError):
    """Error al generar perfil psicológico"""
    pass


class ReportGenerationError(PsychologicalValidationError):
    """Error al generar reporte"""
    pass


class TokenExpiredError(PsychologicalValidationError):
    """Token de acceso expirado"""
    pass


class TokenInvalidError(PsychologicalValidationError):
    """Token de acceso inválido"""
    pass


class PlatformNotSupportedError(PsychologicalValidationError):
    """Plataforma no soportada"""
    pass


class RateLimitExceededError(PsychologicalValidationError):
    """Límite de rate limit excedido"""
    pass


class ConfigurationError(PsychologicalValidationError):
    """Error de configuración"""
    pass




