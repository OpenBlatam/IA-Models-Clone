"""
User Rate Limiter - Rate Limiting por Usuario
==============================================

Sistema de rate limiting granular por usuario/IP.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Regla de rate limiting"""
    max_requests: int
    window_seconds: float
    burst: Optional[int] = None  # Permite burst inicial
    per_user: bool = True  # Si aplicar por usuario o global


@dataclass
class UserRateLimit:
    """Rate limit de un usuario"""
    user_id: str
    requests: deque = field(default_factory=lambda: deque())
    last_reset: float = field(default_factory=time.time)
    blocked_until: Optional[float] = None
    
    def add_request(self, timestamp: float) -> None:
        """Agregar request"""
        self.requests.append(timestamp)
    
    def cleanup_old_requests(self, window_seconds: float) -> None:
        """Limpiar requests fuera de la ventana"""
        cutoff = time.time() - window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    def get_count(self, window_seconds: float) -> int:
        """Obtener número de requests en ventana"""
        self.cleanup_old_requests(window_seconds)
        return len(self.requests)
    
    def is_blocked(self) -> bool:
        """Verificar si está bloqueado"""
        if self.blocked_until:
            if time.time() < self.blocked_until:
                return True
            else:
                self.blocked_until = None
        return False


class UserRateLimiter:
    """
    Rate limiter por usuario/IP.
    
    Permite diferentes límites para diferentes usuarios.
    """
    
    def __init__(self):
        self.user_limits: Dict[str, UserRateLimit] = {}
        self.global_limit: Optional[UserRateLimit] = None
        self.rules: Dict[str, RateLimitRule] = {}
        self.default_rule: Optional[RateLimitRule] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def set_default_rule(
        self,
        max_requests: int,
        window_seconds: float,
        burst: Optional[int] = None
    ) -> None:
        """
        Establecer regla por defecto.
        
        Args:
            max_requests: Máximo de requests
            window_seconds: Ventana de tiempo en segundos
            burst: Permite burst inicial
        """
        self.default_rule = RateLimitRule(
            max_requests=max_requests,
            window_seconds=window_seconds,
            burst=burst
        )
        logger.info(f"🚦 Default rate limit rule set: {max_requests}/{window_seconds}s")
    
    def set_user_rule(
        self,
        user_id: str,
        max_requests: int,
        window_seconds: float,
        burst: Optional[int] = None
    ) -> None:
        """
        Establecer regla para usuario específico.
        
        Args:
            user_id: ID de usuario
            max_requests: Máximo de requests
            window_seconds: Ventana de tiempo
            burst: Permite burst inicial
        """
        self.rules[user_id] = RateLimitRule(
            max_requests=max_requests,
            window_seconds=window_seconds,
            burst=burst,
            per_user=True
        )
        logger.info(f"🚦 Rate limit rule set for user {user_id}: {max_requests}/{window_seconds}s")
    
    async def check_rate_limit(
        self,
        user_id: str,
        rule_name: Optional[str] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Verificar rate limit para usuario.
        
        Args:
            user_id: ID de usuario
            rule_name: Nombre de regla específica (opcional)
            
        Returns:
            Tupla (permitido, tiempo_restante)
        """
        # Obtener regla
        rule = None
        if rule_name and rule_name in self.rules:
            rule = self.rules[rule_name]
        elif user_id in self.rules:
            rule = self.rules[user_id]
        else:
            rule = self.default_rule
        
        if not rule:
            return True, None
        
        # Obtener o crear rate limit para usuario
        if user_id not in self.user_limits:
            self.user_limits[user_id] = UserRateLimit(user_id=user_id)
        
        user_limit = self.user_limits[user_id]
        
        # Verificar si está bloqueado
        if user_limit.is_blocked():
            remaining = user_limit.blocked_until - time.time()
            return False, remaining
        
        # Verificar límite
        current_count = user_limit.get_count(rule.window_seconds)
        
        # Permitir burst inicial
        if rule.burst and current_count == 0:
            user_limit.add_request(time.time())
            return True, None
        
        if current_count >= rule.max_requests:
            # Bloquear temporalmente
            user_limit.blocked_until = time.time() + rule.window_seconds
            remaining = rule.window_seconds
            logger.warning(f"🚦 Rate limit exceeded for user {user_id}")
            return False, remaining
        
        # Permitir request
        user_limit.add_request(time.time())
        return True, None
    
    async def reset_user_limit(self, user_id: str) -> None:
        """
        Resetear rate limit de usuario.
        
        Args:
            user_id: ID de usuario
        """
        if user_id in self.user_limits:
            self.user_limits[user_id].requests.clear()
            self.user_limits[user_id].blocked_until = None
            logger.info(f"🚦 Rate limit reset for user {user_id}")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de rate limit para usuario.
        
        Args:
            user_id: ID de usuario
            
        Returns:
            Diccionario con estadísticas
        """
        if user_id not in self.user_limits:
            return {
                "user_id": user_id,
                "requests_count": 0,
                "is_blocked": False,
                "blocked_until": None
            }
        
        user_limit = self.user_limits[user_id]
        
        # Obtener regla aplicada
        rule = self.rules.get(user_id) or self.default_rule
        window_seconds = rule.window_seconds if rule else 60.0
        
        return {
            "user_id": user_id,
            "requests_count": user_limit.get_count(window_seconds),
            "max_requests": rule.max_requests if rule else None,
            "window_seconds": window_seconds,
            "is_blocked": user_limit.is_blocked(),
            "blocked_until": user_limit.blocked_until,
            "remaining_requests": (
                (rule.max_requests - user_limit.get_count(window_seconds))
                if rule else None
            )
        }
    
    async def start_cleanup(self, interval: float = 300.0) -> None:
        """
        Iniciar limpieza periódica.
        
        Args:
            interval: Intervalo de limpieza en segundos
        """
        if self._cleanup_task and not self._cleanup_task.done():
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval))
        logger.info("🧹 Rate limiter cleanup started")
    
    async def stop_cleanup(self) -> None:
        """Detener limpieza"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self, interval: float) -> None:
        """Loop de limpieza"""
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Limpiar usuarios inactivos (sin requests en última hora)
                cutoff = time.time() - 3600
                inactive_users = [
                    user_id for user_id, user_limit in self.user_limits.items()
                    if not user_limit.requests or user_limit.requests[-1] < cutoff
                ]
                
                for user_id in inactive_users:
                    del self.user_limits[user_id]
                
                if inactive_users:
                    logger.debug(f"🧹 Cleaned up {len(inactive_users)} inactive users")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")




