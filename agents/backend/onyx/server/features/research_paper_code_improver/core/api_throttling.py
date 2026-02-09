"""
API Throttling System - Sistema de throttling para APIs
========================================================
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ThrottleRule:
    """Regla de throttling"""
    identifier: str
    max_requests: int
    window_seconds: int
    burst: Optional[int] = None


@dataclass
class ThrottleResult:
    """Resultado de throttling"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[float] = None


class APIThrottlingSystem:
    """Sistema de throttling para APIs"""
    
    def __init__(self):
        self.rules: Dict[str, ThrottleRule] = {}
        self.request_counts: Dict[str, List[float]] = {}  # identifier -> timestamps
    
    def add_rule(self, rule: ThrottleRule):
        """Agrega una regla de throttling"""
        self.rules[rule.identifier] = rule
        self.request_counts[rule.identifier] = []
    
    def check_throttle(self, identifier: str) -> ThrottleResult:
        """Verifica throttling"""
        rule = self.rules.get(identifier)
        if not rule:
            return ThrottleResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_time=datetime.now()
            )
        
        now = time.time()
        window_start = now - rule.window_seconds
        
        # Limpiar requests antiguos
        counts = self.request_counts[identifier]
        counts[:] = [t for t in counts if t > window_start]
        
        # Verificar límite
        allowed = len(counts) < rule.max_requests
        
        if allowed:
            counts.append(now)
        
        remaining = max(0, rule.max_requests - len(counts))
        reset_time = datetime.now() + timedelta(seconds=rule.window_seconds)
        
        retry_after = None
        if not allowed and counts:
            oldest = min(counts)
            retry_after = rule.window_seconds - (now - oldest)
        
        return ThrottleResult(
            allowed=allowed,
            limit=rule.max_requests,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    def get_throttle_info(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de throttling"""
        rule = self.rules.get(identifier)
        if not rule:
            return None
        
        counts = self.request_counts.get(identifier, [])
        now = time.time()
        window_start = now - rule.window_seconds
        recent_counts = [t for t in counts if t > window_start]
        
        return {
            "identifier": identifier,
            "limit": rule.max_requests,
            "window_seconds": rule.window_seconds,
            "current_count": len(recent_counts),
            "remaining": max(0, rule.max_requests - len(recent_counts))
        }




