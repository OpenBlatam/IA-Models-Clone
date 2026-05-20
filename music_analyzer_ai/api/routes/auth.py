"""
Authentication endpoints
"""

from fastapi import Query, Header
from typing import Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class AuthRouter(BaseRouter):
    """Router for authentication endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/auth", tags=["Authentication"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all auth routes"""
        
        @self.router.post("/register", response_model=dict)
        @self.handle_exceptions
        async def register_user(
            username: str = Query(...),
            email: str = Query(...),
            password: str = Query(...)
        ):
            """Registra un nuevo usuario"""
            auth_service = self.get_service("auth_service")
            result = auth_service.register(username, email, password)
            self.require_success(result, "Error al registrar usuario", status_code=400)
            return self.success_response(result, message="Usuario registrado exitosamente")
        
        @self.router.post("/login", response_model=dict)
        @self.handle_exceptions
        async def login(
            username: str = Query(...),
            password: str = Query(...)
        ):
            """Inicia sesión y obtiene token"""
            auth_service = self.get_service("auth_service")
            result = auth_service.login(username, password)
            self.require_success(result, "Credenciales inválidas", status_code=401)
            return self.success_response(result, message="Login exitoso")
        
        @self.router.get("/me", response_model=dict)
        @self.handle_exceptions
        async def get_current_user(
            authorization: Optional[str] = Header(None)
        ):
            """Obtiene información del usuario actual"""
            token = self.extract_bearer_token(authorization)
            auth_service = self.get_service("auth_service")
            user = auth_service.get_current_user(token)
            self.require_not_none(user, "Token inválido", status_code=401)
            return self.success_response({"user": user})


def get_auth_router() -> AuthRouter:
    """Factory function to get auth router"""
    return AuthRouter()

