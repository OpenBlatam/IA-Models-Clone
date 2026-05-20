"""
Excepciones personalizadas para Dermatology AI
"""


class DermatologyAIException(Exception):
    """Excepción base para el sistema de dermatología"""
    pass


class ImageProcessingError(DermatologyAIException):
    """Error en procesamiento de imagen"""
    pass


class VideoProcessingError(DermatologyAIException):
    """Error en procesamiento de video"""
    pass


class AnalysisError(DermatologyAIException):
    """Error en análisis de piel"""
    pass


class ValidationError(DermatologyAIException):
    """Error de validación"""
    pass


class CacheError(DermatologyAIException):
    """Error en cache"""
    pass






