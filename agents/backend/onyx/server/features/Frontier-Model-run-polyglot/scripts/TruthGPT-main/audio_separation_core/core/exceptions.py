"""
Excepciones Core - Jerarquía de excepciones para Audio Separation Core.

Define todas las excepciones específicas del sistema de separación de audio.
"""

from __future__ import annotations


class AudioSeparationError(Exception):
    """
    Excepción base para todos los errores de separación de audio.
    """
    
    def __init__(
        self,
        message: str,
        component: str = None,
        error_code: str = None,
        details: dict = None
    ):
        super().__init__(message)
        self.message = message
        self.component = component
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.component:
            parts.append(f"Component: {self.component}")
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        return " | ".join(parts)


class AudioProcessingError(AudioSeparationError):
    """
    Error durante el procesamiento de audio.
    """
    pass


class AudioFormatError(AudioSeparationError):
    """
    Error relacionado con formatos de audio no soportados o inválidos.
    """
    pass


class AudioModelError(AudioSeparationError):
    """
    Error relacionado con modelos de IA para separación de audio.
    """
    pass


class AudioValidationError(AudioSeparationError):
    """
    Error de validación de datos o parámetros.
    """
    pass


class AudioIOError(AudioSeparationError):
    """
    Error de lectura/escritura de archivos de audio.
    """
    pass


class AudioInitializationError(AudioSeparationError):
    """
    Error durante la inicialización de componentes.
    """
    pass


class AudioConfigurationError(AudioSeparationError):
    """
    Error de configuración.
    """
    pass




