"""
Manejo centralizado de errores (legacy compatibility).

This module is kept for backward compatibility.
New code should use core.error_handling.UnifiedErrorHandler instead.
"""

from core.error_handling import (
    UnifiedErrorHandler,
    ErrorCategory,
    error_handler
)

# Legacy ErrorHandler class for backward compatibility
class ErrorHandler:
    """Manejo centralizado de errores (legacy)"""
    
    @staticmethod
    def handle_generation_error(error: Exception, context=None):
        """Maneja errores durante la generación"""
        return UnifiedErrorHandler.handle_generation_error(error, context)
    
    @staticmethod
    def handle_audio_processing_error(error: Exception, context=None):
        """Maneja errores durante el procesamiento de audio"""
        return UnifiedErrorHandler.handle_audio_processing_error(error, context)
    
    @staticmethod
    def handle_validation_error(error: ValueError, context=None):
        """Maneja errores de validación"""
        return UnifiedErrorHandler.handle_validation_error(error, context)
    
    @staticmethod
    def handle_cache_error(error: Exception, context=None):
        """Maneja errores de caché de forma silenciosa"""
        return UnifiedErrorHandler.handle_cache_error(error, context)

