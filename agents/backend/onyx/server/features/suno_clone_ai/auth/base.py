"""
Base Authenticator - Clase base para autenticadores
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseAuthenticator(ABC):
    """Clase base abstracta para autenticadores"""

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Autentica un usuario"""
        pass

    @abstractmethod
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token"""
        pass

