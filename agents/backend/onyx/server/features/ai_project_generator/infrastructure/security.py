"""
Security Service - Servicio de seguridad
========================================

Servicio de seguridad que abstrae OAuth2 y otras funcionalidades de seguridad.
"""

import logging
from typing import Optional

from ..core.oauth2_security import get_oauth2_manager, get_current_user

logger = logging.getLogger(__name__)


class SecurityService:
    """Servicio de seguridad"""
    
    def __init__(self):
        self.oauth2_manager = get_oauth2_manager()
    
    def get_current_user_dependency(self):
        """Obtiene dependency para usuario actual"""
        return get_current_user
    
    def create_access_token(self, data: dict) -> str:
        """Crea token de acceso"""
        return self.oauth2_manager.create_access_token(data)
    
    def verify_token(self, token: str):
        """Verifica token"""
        return self.oauth2_manager.verify_token(token)















