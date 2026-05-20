"""
Rate Limit Service - Servicio de rate limiting para GitHub API.
"""

import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from config.logging_config import get_logger
from core.exceptions import GitHubAgentError

logger = get_logger(__name__)


class RateLimitExceededError(GitHubAgentError):
    """Error cuando se excede el rate limit."""

    def __init__(
        self,
        message: str,
        reset_time: Optional[datetime] = None,
        retry_after: Optional[int] = None
    ):
        """
        Inicializar error de rate limit.

        Args:
            message: Mensaje de error
            reset_time: Tiempo cuando se resetea el rate limit
            retry_after: Segundos hasta que se puede reintentar
        """
        super().__init__(message)
        self.reset_time = reset_time
        self.retry_after = retry_after


class RateLimitService:
    """
    Servicio de rate limiting para GitHub API con mejoras.
    
    Attributes:
        limit: Límite de requests por ventana
        window_seconds: Duración de la ventana en segundos
        requests: Diccionario de requests por identificador
        blocked_until: Diccionario de bloqueos por identificador
    """

    # Límites de GitHub API (requests por hora)
    GITHUB_RATE_LIMIT_AUTHENTICATED = 5000
    GITHUB_RATE_LIMIT_UNAUTHENTICATED = 60

    def __init__(
        self,
        limit: int = GITHUB_RATE_LIMIT_AUTHENTICATED,
        window_seconds: int = 3600  # 1 hora
    ):
        """
        Inicializar servicio de rate limiting con validaciones.

        Args:
            limit: Límite de requests por ventana (debe ser entero positivo)
            window_seconds: Duración de la ventana en segundos (debe ser entero positivo)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit debe ser un entero positivo, recibido: {limit}")
        
        if not isinstance(window_seconds, int) or window_seconds < 1:
            raise ValueError(f"window_seconds debe ser un entero positivo, recibido: {window_seconds}")
        
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self.blocked_until: Dict[str, datetime] = {}
        
        logger.info(
            f"✅ RateLimitService inicializado: limit={limit}, window={window_seconds}s"
        )

    def check_rate_limit(
        self,
        identifier: str,
        cost: int = 1
    ) -> bool:
        """
        Verificar si se puede hacer un request con validaciones.

        Args:
            identifier: Identificador único (token, IP, etc.) - debe ser string no vacío
            cost: Costo del request (algunos endpoints cuentan más) - debe ser entero positivo

        Returns:
            True si se puede hacer el request

        Raises:
            RateLimitExceededError: Si se excedió el límite
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not identifier or not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"identifier debe ser un string no vacío, recibido: {identifier}")
        
        if not isinstance(cost, int) or cost < 1:
            raise ValueError(f"cost debe ser un entero positivo, recibido: {cost}")
        
        identifier = identifier.strip()
        
        now = datetime.now()

        # Verificar si está bloqueado
        if identifier in self.blocked_until:
            if now < self.blocked_until[identifier]:
                remaining = (self.blocked_until[identifier] - now).total_seconds()
                logger.warning(
                    f"⚠️  Rate limit excedido para {identifier}. "
                    f"Espera {int(remaining)} segundos."
                )
                raise RateLimitExceededError(
                    f"Rate limit excedido para {identifier}. "
                    f"Espera {int(remaining)} segundos.",
                    reset_time=self.blocked_until[identifier],
                    retry_after=int(remaining)
                )
            else:
                # Bloqueo expirado
                del self.blocked_until[identifier]
                logger.debug(f"Bloqueo expirado para {identifier}")

        # Limpiar requests antiguos fuera de la ventana
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        before_cleanup = len(self.requests[identifier])
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > cutoff_time
        ]
        after_cleanup = len(self.requests[identifier])
        
        if before_cleanup != after_cleanup:
            logger.debug(
                f"Limpiados {before_cleanup - after_cleanup} requests antiguos "
                f"para {identifier}"
            )

        # Verificar límite
        current_count = len(self.requests[identifier])
        if current_count + cost > self.limit:
            reset_time = now + timedelta(seconds=self.window_seconds)
            self.blocked_until[identifier] = reset_time
            logger.warning(
                f"❌ Rate limit excedido para {identifier}. "
                f"Límite: {self.limit}, Usado: {current_count}, Cost: {cost}"
            )
            raise RateLimitExceededError(
                f"Rate limit excedido para {identifier}. "
                f"Límite: {self.limit}, Usado: {current_count}, Cost: {cost}",
                reset_time=reset_time,
                retry_after=self.window_seconds
            )

        # Registrar request
        self.requests[identifier].append(now)
        remaining = self.limit - (current_count + cost)
        logger.debug(
            f"✅ Rate limit OK para {identifier}: "
            f"{current_count + cost}/{self.limit} (remaining: {remaining})"
        )
        return True

    def record_request(
        self,
        identifier: str,
        cost: int = 1
    ) -> None:
        """
        Registrar un request (sin verificar límite) con validaciones.

        Args:
            identifier: Identificador único (debe ser string no vacío)
            cost: Costo del request (debe ser entero positivo)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not identifier or not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"identifier debe ser un string no vacío, recibido: {identifier}")
        
        if not isinstance(cost, int) or cost < 1:
            raise ValueError(f"cost debe ser un entero positivo, recibido: {cost}")
        
        identifier = identifier.strip()
        
        now = datetime.now()
        for _ in range(cost):
            self.requests[identifier].append(now)
        
        logger.debug(f"Request registrado para {identifier} (cost: {cost})")

    def get_remaining(
        self,
        identifier: str
    ) -> int:
        """
        Obtener requests restantes con validaciones.

        Args:
            identifier: Identificador único (debe ser string no vacío)

        Returns:
            Número de requests restantes
            
        Raises:
            ValueError: Si identifier es inválido
        """
        # Validación
        if not identifier or not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"identifier debe ser un string no vacío, recibido: {identifier}")
        
        identifier = identifier.strip()
        
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > cutoff_time
        ]
        current_count = len(self.requests[identifier])
        remaining = max(0, self.limit - current_count)
        
        logger.debug(
            f"Requests restantes para {identifier}: {remaining} "
            f"({current_count}/{self.limit} usado)"
        )
        
        return remaining

    def get_reset_time(
        self,
        identifier: str
    ) -> Optional[datetime]:
        """
        Obtener tiempo de reset del rate limit.

        Args:
            identifier: Identificador único

        Returns:
            Tiempo de reset o None si no está bloqueado
        """
        if identifier in self.blocked_until:
            return self.blocked_until[identifier]
        return None

    def reset(self, identifier: Optional[str] = None) -> None:
        """
        Resetear rate limit para un identificador o todos.

        Args:
            identifier: Identificador a resetear (None para todos)
        """
        if identifier:
            self.requests.pop(identifier, None)
            self.blocked_until.pop(identifier, None)
            logger.info(f"Rate limit reseteado para {identifier}")
        else:
            self.requests.clear()
            self.blocked_until.clear()
            logger.info("Rate limit reseteado para todos los identificadores")

    def get_stats(self, identifier: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de rate limit.

        Args:
            identifier: Identificador único

        Returns:
            Diccionario con estadísticas
        """
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > cutoff_time
        ]

        current_count = len(self.requests[identifier])
        remaining = max(0, self.limit - current_count)
        reset_time = self.get_reset_time(identifier)

        return {
            "limit": self.limit,
            "used": current_count,
            "remaining": remaining,
            "window_seconds": self.window_seconds,
            "reset_time": reset_time.isoformat() if reset_time else None,
            "blocked": identifier in self.blocked_until
        }

