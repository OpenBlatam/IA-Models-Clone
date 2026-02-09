"""
Excepciones personalizadas para la API

Sigue el patrón de error handling consistente con mensajes user-friendly.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Excepción base para todas las excepciones de la API"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class SongNotFoundError(BaseAPIException):
    """Excepción cuando una canción no se encuentra"""
    
    def __init__(self, song_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Song with ID '{song_id}' not found"
        )


class SongGenerationError(BaseAPIException):
    """Excepción cuando falla la generación de una canción"""
    
    def __init__(self, message: str, song_id: Optional[str] = None):
        detail = f"Failed to generate song: {message}"
        if song_id:
            detail += f" (song_id: {song_id})"
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class AudioProcessingError(BaseAPIException):
    """Excepción cuando falla el procesamiento de audio"""
    
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {message}"
        )


class AudioFileNotFoundError(BaseAPIException):
    """Excepción cuando no se encuentra un archivo de audio"""
    
    def __init__(self, file_path: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found: {file_path}"
        )


class InvalidInputError(BaseAPIException):
    """Excepción para inputs inválidos"""
    
    def __init__(self, message: str, field: Optional[str] = None):
        detail = message
        if field:
            detail = f"Invalid {field}: {message}"
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )


class RateLimitError(BaseAPIException):
    """Excepción cuando se excede el rate limit"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            headers={"Retry-After": "60"}
        )


class CacheError(BaseAPIException):
    """Excepción relacionada con el caché"""
    
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache error: {message}"
        )


class DatabaseError(BaseAPIException):
    """Excepción relacionada con la base de datos"""
    
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {message}"
        )


class ValidationError(BaseAPIException):
    """Excepción para errores de validación"""
    
    def __init__(self, message: str, errors: Optional[Dict[str, Any]] = None):
        detail = {"message": message}
        if errors:
            detail["errors"] = errors
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )

