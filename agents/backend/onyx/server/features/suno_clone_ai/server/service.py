"""
Server Service - Servicio del servidor
"""

from typing import Optional
from fastapi import FastAPI
from .app import create_app
from .routes import setup_routes
from auth.service import AuthService
from access.service import AccessService
from chat.service import ChatService
from tracing.service import TracingService


class ServerService:
    """Servicio para gestionar el servidor"""

    def __init__(
        self,
        auth_service: Optional[AuthService] = None,
        access_service: Optional[AccessService] = None,
        chat_service: Optional[ChatService] = None,
        tracing_service: Optional[TracingService] = None
    ):
        """Inicializa el servicio del servidor"""
        self.app = create_app()
        self.auth_service = auth_service
        self.access_service = access_service
        self.chat_service = chat_service
        self.tracing_service = tracing_service
        setup_routes(self.app, self)

    def get_app(self) -> FastAPI:
        """Obtiene la aplicación FastAPI"""
        return self.app

