"""
Custom Exceptions - Excepciones personalizadas
=============================================

Excepciones específicas del dominio para mejor manejo de errores.
"""


class AIProjectGeneratorError(Exception):
    """Excepción base para el generador de proyectos"""
    pass


class ProjectNotFoundError(AIProjectGeneratorError):
    """Proyecto no encontrado"""
    pass


class ProjectGenerationError(AIProjectGeneratorError):
    """Error en generación de proyecto"""
    pass


class ValidationError(AIProjectGeneratorError):
    """Error de validación"""
    pass


class CacheError(AIProjectGeneratorError):
    """Error de cache"""
    pass


class ServiceUnavailableError(AIProjectGeneratorError):
    """Servicio no disponible"""
    pass


class ConfigurationError(AIProjectGeneratorError):
    """Error de configuración"""
    pass


class RepositoryError(AIProjectGeneratorError):
    """Error en repositorio"""
    pass















