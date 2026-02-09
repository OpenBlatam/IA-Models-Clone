"""
Throttle Manager System
=======================

Sistema de throttling para control de velocidad.
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ThrottleRule:
    """Regla de throttling."""
    rule_id: str
    key: str
    max_requests: int
    per_second: float
    burst: int = 0  # Permite ráfagas
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThrottleManager:
    """
    Gestor de throttling.
    
    Controla la velocidad de requests.
    """
    
    def __init__(self):
        """Inicializar gestor de throttling."""
        self.rules: Dict[str, ThrottleRule] = {}
        self.request_times: Dict[str, deque] = defaultdict(deque)
        self.burst_tokens: Dict[str, int] = defaultdict(int)
    
    def add_rule(
        self,
        rule_id: str,
        key: str,
        max_requests: int,
        per_second: float,
        burst: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThrottleRule:
        """
        Agregar regla de throttling.
        
        Args:
            rule_id: ID único de la regla
            key: Clave para identificar el throttle
            max_requests: Número máximo de requests
            per_second: Requests por segundo
            burst: Tokens de ráfaga
            metadata: Metadata adicional
            
        Returns:
            Regla creada
        """
        rule = ThrottleRule(
            rule_id=rule_id,
            key=key,
            max_requests=max_requests,
            per_second=per_second,
            burst=burst,
            metadata=metadata or {}
        )
        
        self.rules[rule_id] = rule
        
        # Inicializar tokens de ráfaga
        if burst > 0:
            self.burst_tokens[f"{key}"] = burst
        
        logger.info(f"Added throttle rule: {rule_id}")
        
        return rule
    
    async def throttle(
        self,
        rule_id: str,
        identifier: str
    ) -> float:
        """
        Aplicar throttling.
        
        Args:
            rule_id: ID de la regla
            identifier: Identificador
            
        Returns:
            Tiempo de espera en segundos (0 si no hay que esperar)
        """
        if rule_id not in self.rules:
            return 0.0
        
        rule = self.rules[rule_id]
        key = f"{rule.key}:{identifier}"
        now = time.time()
        
        # Limpiar timestamps antiguos
        request_times = self.request_times[key]
        cutoff = now - 1.0  # Último segundo
        while request_times and request_times[0] < cutoff:
            request_times.popleft()
        
        # Verificar límite de requests por segundo
        if len(request_times) >= rule.max_requests:
            # Calcular tiempo de espera
            oldest_request = request_times[0]
            wait_time = oldest_request + 1.0 - now
            return max(0.0, wait_time)
        
        # Verificar burst tokens
        burst_key = f"{rule.key}:{identifier}"
        if rule.burst > 0:
            if self.burst_tokens[burst_key] > 0:
                self.burst_tokens[burst_key] -= 1
                request_times.append(now)
                return 0.0
        
        # Agregar request
        request_times.append(now)
        
        # Refill burst tokens
        if rule.burst > 0:
            refill_rate = rule.burst / rule.per_second
            elapsed = now - (request_times[0] if request_times else now)
            if elapsed > 0:
                refill = int(elapsed * refill_rate)
                self.burst_tokens[burst_key] = min(
                    rule.burst,
                    self.burst_tokens[burst_key] + refill
                )
        
        return 0.0
    
    def get_rule(self, rule_id: str) -> Optional[ThrottleRule]:
        """Obtener regla por ID."""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[ThrottleRule]:
        """Listar todas las reglas."""
        return list(self.rules.values())


# Instancia global
_throttle_manager: Optional[ThrottleManager] = None


def get_throttle_manager() -> ThrottleManager:
    """Obtener instancia global del gestor de throttling."""
    global _throttle_manager
    if _throttle_manager is None:
        _throttle_manager = ThrottleManager()
    return _throttle_manager


# Importar defaultdict
from collections import defaultdict






