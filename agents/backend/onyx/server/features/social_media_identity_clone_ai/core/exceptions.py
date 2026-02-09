"""
Excepciones personalizadas para el sistema
"""

from typing import Optional, Dict, Any


class SocialMediaIdentityCloneError(Exception):
    """Excepción base para el sistema"""
    
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


class ProfileExtractionError(SocialMediaIdentityCloneError):
    """Error al extraer perfil de red social"""
    pass


class IdentityAnalysisError(SocialMediaIdentityCloneError):
    """Error al analizar identidad"""
    pass


class ContentGenerationError(SocialMediaIdentityCloneError):
    """Error al generar contenido"""
    pass


class ModelLoadingError(SocialMediaIdentityCloneError):
    """Error al cargar modelo de ML"""
    pass


class TrainingError(SocialMediaIdentityCloneError):
    """Error durante el entrenamiento"""
    pass


class InferenceError(SocialMediaIdentityCloneError):
    """Error durante la inferencia"""
    pass


class ValidationError(SocialMediaIdentityCloneError):
    """Error de validación"""
    pass


class CacheError(SocialMediaIdentityCloneError):
    """Error en el sistema de caché"""
    pass


class ConnectorError(SocialMediaIdentityCloneError):
    """Error en conectores de redes sociales"""
    pass




