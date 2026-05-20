"""
Excepciones personalizadas para la comunidad Lovable (optimizado)

Sigue el patrón de error handling consistente con mensajes user-friendly.

Todas las excepciones heredan de BaseCommunityException que a su vez hereda de HTTPException,
permitiendo un manejo consistente de errores en toda la aplicación.

Excepciones disponibles:
- ChatNotFoundError: Cuando un chat no se encuentra (404)
- InvalidChatError: Cuando los datos del chat son inválidos (400)
- DuplicateVoteError: Cuando se intenta votar dos veces con el mismo tipo (400)
- RemixError: Cuando falla la creación de un remix (500)
- ValidationError: Cuando hay errores de validación (422)
- DatabaseError: Cuando hay errores de base de datos (500)
- UnauthorizedError: Cuando el usuario no está autorizado (401)
- ForbiddenError: Cuando el usuario no tiene permisos (403)
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class BaseCommunityException(HTTPException):
    """Excepción base para todas las excepciones de la comunidad"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class ChatNotFoundError(BaseCommunityException):
    """Excepción cuando un chat no se encuentra"""
    
    def __init__(self, chat_id: str, additional_context: Optional[str] = None):
        detail = f"Chat with ID '{chat_id}' not found"
        if additional_context:
            detail += f". {additional_context}"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )
        self.chat_id = chat_id


class InvalidChatError(BaseCommunityException):
    """Excepción para chats inválidos"""
    
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chat: {message}"
        )


class DuplicateVoteError(BaseCommunityException):
    """Excepción cuando se intenta votar dos veces con el mismo tipo"""
    
    def __init__(self, chat_id: str, vote_type: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User has already {vote_type}d chat '{chat_id}'"
        )


class RemixError(BaseCommunityException):
    """Excepción cuando falla un remix"""
    
    def __init__(self, message: str, original_chat_id: Optional[str] = None):
        detail = f"Failed to remix chat: {message}"
        if original_chat_id:
            detail += f" (original_chat_id: {original_chat_id})"
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class ValidationError(BaseCommunityException):
    """Excepción para errores de validación"""
    
    def __init__(self, message: str, errors: Optional[Dict[str, Any]] = None):
        detail = {"message": message}
        if errors:
            detail["errors"] = errors
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )


class DatabaseError(BaseCommunityException):
    """Excepción relacionada con la base de datos"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        detail = f"Database error: {message}"
        if operation:
            detail = f"Database error during {operation}: {message}"
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )
        self.operation = operation


class UnauthorizedError(BaseCommunityException):
    """Excepción cuando el usuario no está autorizado"""
    
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )


class ForbiddenError(BaseCommunityException):
    """Excepción cuando el usuario no tiene permisos"""
    
    def __init__(self, message: str = "Forbidden"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message
        )


class RateLimitError(BaseCommunityException):
    """Excepción cuando se excede el límite de rate"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            headers=headers
        )

