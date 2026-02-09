"""
Policy Manager - Gestión de políticas de acceso
"""
from typing import Optional, Dict, Any, Callable


class PolicyManager:
    """Gestor de políticas de acceso"""
    
    def __init__(self):
        self.policies: Dict[str, Callable] = {}
    
    async def evaluate(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Evalúa una política"""
        policy_key = f"{resource}:{action}"
        policy = self.policies.get(policy_key)
        
        if policy is None:
            return True  # Por defecto permitir si no hay política
        
        return await policy(user_id, resource, action, context or {})
    
    def register_policy(
        self,
        resource: str,
        action: str,
        policy_func: Callable
    ):
        """Registra una política"""
        policy_key = f"{resource}:{action}"
        self.policies[policy_key] = policy_func

