"""
Access Policies - Políticas de acceso
"""

from typing import Dict, Any, Callable
from .models import Permission, Role


class AccessPolicy:
    """Política de acceso"""

    def __init__(self):
        """Inicializa la política de acceso"""
        self._policies: Dict[str, Callable] = {}

    def register_policy(self, resource: str, policy: Callable) -> None:
        """Registra una política para un recurso"""
        self._policies[resource] = policy

    def check(self, user_id: str, resource: str, action: str) -> bool:
        """Verifica una política"""
        policy = self._policies.get(resource)
        if policy:
            return policy(user_id, action)
        return False

