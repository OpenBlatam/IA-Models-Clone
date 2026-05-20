#!/usr/bin/env python3
"""
Model Guard - Protección del modelo contra extracción de información y uso no autorizado.
Basado en papers: "Extracting Training Data from Large Language Models" (Carlini et al., 2021),
"Membership Inference Attacks Against Language Models" (2022).
"""

import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class GuardAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    LOG = "log"
    RATE_LIMIT = "rate_limit"

@dataclass
class RequestContext:
    user_id: str
    endpoint: str
    prompt: str
    timestamp: float
    session_id: Optional[str] = None

class ModelGuard:
    """Guarda de seguridad para proteger el modelo contra abusos."""

    def __init__(self):
        self._rate_limits: Dict[str, List[float]] = {}
        self._blocked_users: set = set()
        self._allowed_ips: set = set()
        self.stats = {"allowed": 0, "blocked": 0, "rate_limited": 0}

    def check_request(self, ctx: RequestContext) -> Dict[str, Any]:
        """Evalúa si la solicitud debe ser permitida."""
        # Verificar si el usuario está bloqueado
        if ctx.user_id in self._blocked_users:
            self.stats["blocked"] += 1
            return {"action": GuardAction.BLOCK.value, "reason": "user_blocked"}

        # Rate limiting
        max_requests_per_minute = 60
        now = time.time()
        if ctx.user_id not in self._rate_limits:
            self._rate_limits[ctx.user_id] = []
        timestamps = self._rate_limits[ctx.user_id]
        # Limpiar timestamps viejos
        timestamps = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= max_requests_per_minute:
            self.stats["rate_limited"] += 1
            return {"action": GuardAction.RATE_LIMIT.value, "reason": "rate_limit_exceeded"}
        timestamps.append(now)
        self._rate_limits[ctx.user_id] = timestamps

        # Verificar hash del prompt para detectar repeticiones exactas (ataque de extracción)
        prompt_hash = hashlib.sha256(ctx.prompt.encode()).hexdigest()
        # (En producción, guardar historial y detectar)

        self.stats["allowed"] += 1
        return {"action": GuardAction.ALLOW.value, "reason": "ok"}

    def block_user(self, user_id: str):
        self._blocked_users.add(user_id)
        logger.warning(f"User {user_id} blocked.")

    def unblock_user(self, user_id: str):
        self._blocked_users.discard(user_id)

    def get_stats(self) -> dict:
        return dict(self.stats)