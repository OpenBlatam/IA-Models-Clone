"""
Advanced Security - Seguridad Avanzada
======================================

Sistema de seguridad avanzado con detección de amenazas, análisis de comportamiento y protección.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Nivel de amenaza."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Tipo de evento de seguridad."""
    AUTH_FAILURE = "auth_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    MALICIOUS_PAYLOAD = "malicious_payload"


@dataclass
class SecurityEvent:
    """Evento de seguridad."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False


@dataclass
class SecurityRule:
    """Regla de seguridad."""
    rule_id: str
    name: str
    pattern: str
    action: str  # block, alert, log
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedSecurity:
    """Sistema de seguridad avanzado."""
    
    def __init__(
        self,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 15,
        threat_threshold: float = 0.7,
    ):
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        self.threat_threshold = threat_threshold
        self.security_events: List[SecurityEvent] = []
        self.security_rules: Dict[str, SecurityRule] = {}
        self.failed_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.locked_accounts: Dict[str, datetime] = {}
        self.threat_scores: Dict[str, float] = defaultdict(float)
        self.behavioral_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = asyncio.Lock()
    
    def add_security_rule(
        self,
        rule_id: str,
        name: str,
        pattern: str,
        action: str = "alert",
        enabled: bool = True,
    ):
        """Agregar regla de seguridad."""
        rule = SecurityRule(
            rule_id=rule_id,
            name=name,
            pattern=pattern,
            action=action,
            enabled=enabled,
        )
        
        self.security_rules[rule_id] = rule
        logger.info(f"Added security rule: {rule_id} - {name}")
    
    async def record_event(
        self,
        event_type: SecurityEventType,
        source: str,
        description: str,
        threat_level: Optional[ThreatLevel] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        """
        Registrar evento de seguridad.
        
        Args:
            event_type: Tipo de evento
            source: Origen del evento
            description: Descripción
            threat_level: Nivel de amenaza (se calcula si no se proporciona)
            metadata: Metadatos adicionales
        
        Returns:
            Evento de seguridad creado
        """
        if threat_level is None:
            threat_level = self._calculate_threat_level(event_type, source)
        
        event = SecurityEvent(
            event_id=f"sec_{source}_{datetime.now().timestamp()}",
            event_type=event_type,
            threat_level=threat_level,
            source=source,
            description=description,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.security_events.append(event)
            
            # Actualizar puntuación de amenaza
            self.threat_scores[source] += self._get_threat_score_increment(threat_level)
            
            # Registrar en patrones de comportamiento
            self.behavioral_patterns[source].append({
                "event_type": event_type.value,
                "timestamp": datetime.now(),
                "threat_level": threat_level.value,
            })
            
            # Aplicar acciones de reglas
            await self._apply_security_rules(event)
            
            # Mantener solo últimos 10000 eventos
            if len(self.security_events) > 10000:
                self.security_events.pop(0)
        
        logger.warning(f"Security event: {event_type.value} from {source} - {threat_level.value}")
        return event
    
    def _calculate_threat_level(
        self,
        event_type: SecurityEventType,
        source: str,
    ) -> ThreatLevel:
        """Calcular nivel de amenaza."""
        base_scores = {
            SecurityEventType.AUTH_FAILURE: 0.2,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 0.4,
            SecurityEventType.RATE_LIMIT_EXCEEDED: 0.3,
            SecurityEventType.UNAUTHORIZED_ACCESS: 0.8,
            SecurityEventType.DATA_BREACH: 0.9,
            SecurityEventType.MALICIOUS_PAYLOAD: 0.95,
        }
        
        base_score = base_scores.get(event_type, 0.1)
        historical_score = self.threat_scores.get(source, 0.0) / 100.0
        
        total_score = base_score + historical_score * 0.3
        
        if total_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif total_score >= 0.6:
            return ThreatLevel.HIGH
        elif total_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _get_threat_score_increment(self, threat_level: ThreatLevel) -> float:
        """Obtener incremento de puntuación de amenaza."""
        increments = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 5.0,
            ThreatLevel.HIGH: 15.0,
            ThreatLevel.CRITICAL: 30.0,
        }
        return increments.get(threat_level, 1.0)
    
    async def _apply_security_rules(self, event: SecurityEvent):
        """Aplicar reglas de seguridad."""
        for rule in self.security_rules.values():
            if not rule.enabled:
                continue
            
            # Verificar si el patrón coincide (simplificado)
            if rule.pattern in event.description.lower():
                if rule.action == "block":
                    event.blocked = True
                    await self._block_source(event.source)
                elif rule.action == "alert":
                    logger.warning(f"Security alert: {rule.name} - {event.description}")
    
    async def _block_source(self, source: str):
        """Bloquear fuente."""
        self.locked_accounts[source] = datetime.now() + timedelta(
            minutes=self.lockout_duration_minutes
        )
        logger.warning(f"Blocked source: {source}")
    
    async def record_failed_auth(self, source: str, reason: str = "Invalid credentials"):
        """Registrar intento de autenticación fallido."""
        async with self._lock:
            self.failed_attempts[source].append({
                "timestamp": datetime.now(),
                "reason": reason,
            })
        
        # Verificar si excede el límite
        recent_attempts = [
            a for a in self.failed_attempts[source]
            if (datetime.now() - a["timestamp"]).total_seconds() < 3600
        ]
        
        if len(recent_attempts) >= self.max_failed_attempts:
            await self.record_event(
                event_type=SecurityEventType.AUTH_FAILURE,
                source=source,
                description=f"Multiple failed auth attempts: {len(recent_attempts)}",
                threat_level=ThreatLevel.HIGH,
            )
            await self._block_source(source)
    
    def is_blocked(self, source: str) -> bool:
        """Verificar si una fuente está bloqueada."""
        lockout_time = self.locked_accounts.get(source)
        if not lockout_time:
            return False
        
        if datetime.now() > lockout_time:
            # Desbloquear automáticamente
            del self.locked_accounts[source]
            return False
        
        return True
    
    def get_threat_score(self, source: str) -> float:
        """Obtener puntuación de amenaza de una fuente."""
        return self.threat_scores.get(source, 0.0)
    
    def get_security_events(
        self,
        source: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        threat_level: Optional[ThreatLevel] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener eventos de seguridad."""
        events = self.security_events
        
        if source:
            events = [e for e in events if e.source == source]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "threat_level": e.threat_level.value,
                "source": e.source,
                "description": e.description,
                "timestamp": e.timestamp.isoformat(),
                "blocked": e.blocked,
                "metadata": e.metadata,
            }
            for e in events[-limit:]
        ]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Obtener resumen de seguridad."""
        by_type: Dict[str, int] = defaultdict(int)
        by_level: Dict[str, int] = defaultdict(int)
        
        for event in self.security_events:
            by_type[event.event_type.value] += 1
            by_level[event.threat_level.value] += 1
        
        return {
            "total_events": len(self.security_events),
            "events_by_type": dict(by_type),
            "events_by_level": dict(by_level),
            "blocked_sources": len(self.locked_accounts),
            "active_rules": sum(1 for r in self.security_rules.values() if r.enabled),
            "monitored_sources": len(self.behavioral_patterns),
        }
    
    async def analyze_behavior(self, source: str) -> Dict[str, Any]:
        """Analizar comportamiento de una fuente."""
        patterns = self.behavioral_patterns.get(source, deque())
        
        if not patterns:
            return {
                "source": source,
                "analysis": "No data available",
            }
        
        recent_patterns = list(patterns)[-100:]
        
        # Analizar frecuencia
        event_counts: Dict[str, int] = defaultdict(int)
        for pattern in recent_patterns:
            event_counts[pattern["event_type"]] += 1
        
        # Calcular anomalías
        total_events = len(recent_patterns)
        anomaly_score = 0.0
        
        if total_events > 10:
            # Si hay muchos eventos de un tipo específico, es anómalo
            for count in event_counts.values():
                if count / total_events > 0.5:
                    anomaly_score += 0.3
        
        return {
            "source": source,
            "total_events": total_events,
            "event_distribution": dict(event_counts),
            "threat_score": self.threat_scores.get(source, 0.0),
            "anomaly_score": min(1.0, anomaly_score),
            "is_blocked": self.is_blocked(source),
        }
















