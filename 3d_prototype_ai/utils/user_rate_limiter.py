"""
User Rate Limiter - Sistema de rate limiting por usuario avanzado
===================================================================
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class UserRateLimiter:
    """Sistema de rate limiting por usuario"""
    
    def __init__(self):
        self.user_limits: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.user_usage: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.user_quotas: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.blocked_users: Dict[str, datetime] = {}
    
    def set_user_limit(self, user_id: str, endpoint: str, requests: int, window_seconds: int = 60):
        """Establece límite para un usuario en un endpoint"""
        key = f"{endpoint}:{window_seconds}"
        self.user_limits[user_id][key] = {
            "requests": requests,
            "window_seconds": window_seconds
        }
    
    def set_user_quota(self, user_id: str, quota_type: str, quota: int):
        """Establece cuota para un usuario"""
        self.user_quotas[user_id][quota_type] = quota
    
    def check_user_limit(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Verifica límite de usuario"""
        # Verificar si está bloqueado
        if user_id in self.blocked_users:
            block_until = self.blocked_users[user_id]
            if datetime.now() < block_until:
                remaining = int((block_until - datetime.now()).total_seconds())
                return {
                    "allowed": False,
                    "reason": "user_blocked",
                    "retry_after": remaining
                }
            else:
                del self.blocked_users[user_id]
        
        # Buscar límite aplicable
        limits = self.user_limits.get(user_id, {})
        for key, limit_config in limits.items():
            if endpoint in key:
                window_seconds = limit_config["window_seconds"]
                requests_limit = limit_config["requests"]
                
                usage_key = f"{endpoint}:{window_seconds}"
                usage = self.user_usage[user_id][usage_key]
                
                # Limpiar entradas fuera de la ventana
                cutoff = datetime.now() - timedelta(seconds=window_seconds)
                while usage and usage[0] < cutoff:
                    usage.popleft()
                
                if len(usage) >= requests_limit:
                    return {
                        "allowed": False,
                        "reason": "rate_limit_exceeded",
                        "limit": requests_limit,
                        "used": len(usage),
                        "reset_in": int((usage[0] + timedelta(seconds=window_seconds) - datetime.now()).total_seconds()) if usage else window_seconds
                    }
                
                # Registrar uso
                usage.append(datetime.now())
                
                return {
                    "allowed": True,
                    "remaining": requests_limit - len(usage),
                    "limit": requests_limit
                }
        
        # Sin límite específico, permitir
        return {"allowed": True, "unlimited": True}
    
    def check_user_quota(self, user_id: str, quota_type: str, amount: int = 1) -> Dict[str, Any]:
        """Verifica cuota de usuario"""
        quotas = self.user_quotas.get(user_id, {})
        quota = quotas.get(quota_type)
        
        if quota is None:
            return {"allowed": True, "unlimited": True}
        
        # Calcular uso actual (simplificado)
        usage_key = f"{quota_type}_usage"
        current_usage = quotas.get(usage_key, 0)
        
        if current_usage + amount > quota:
            return {
                "allowed": False,
                "quota": quota,
                "used": current_usage,
                "requested": amount
            }
        
        # Actualizar uso
        quotas[usage_key] = current_usage + amount
        
        return {
            "allowed": True,
            "quota": quota,
            "used": current_usage + amount,
            "remaining": quota - (current_usage + amount)
        }
    
    def block_user(self, user_id: str, duration_seconds: int = 3600):
        """Bloquea un usuario temporalmente"""
        self.blocked_users[user_id] = datetime.now() + timedelta(seconds=duration_seconds)
        logger.warning(f"Usuario {user_id} bloqueado por {duration_seconds} segundos")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de usuario"""
        limits = self.user_limits.get(user_id, {})
        quotas = self.user_quotas.get(user_id, {})
        usage = self.user_usage.get(user_id, {})
        
        return {
            "user_id": user_id,
            "limits": {k: v for k, v in limits.items()},
            "quotas": {k: v for k, v in quotas.items() if not k.endswith("_usage")},
            "usage": {k: len(v) for k, v in usage.items()},
            "blocked": user_id in self.blocked_users
        }




