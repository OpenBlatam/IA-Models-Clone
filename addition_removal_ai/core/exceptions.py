"""
Exceptions - Excepciones personalizadas del sistema
"""


class AdditionRemovalAIError(Exception):
    """Excepción base del sistema"""
    pass


class ContentValidationError(AdditionRemovalAIError):
    """Error de validación de contenido"""
    pass


class FormatNotSupportedError(AdditionRemovalAIError):
    """Error cuando el formato no es soportado"""
    pass


class AIEngineError(AdditionRemovalAIError):
    """Error en el motor de IA"""
    pass


class PositionError(AdditionRemovalAIError):
    """Error en el posicionamiento"""
    pass


class CacheError(AdditionRemovalAIError):
    """Error en el sistema de cache"""
    pass


class HistoryError(AdditionRemovalAIError):
    """Error en el historial"""
    pass


class BatchOperationError(AdditionRemovalAIError):
    """Error en operación batch"""
    pass






