"""
Security Analyzer - Analizador de Seguridad
==========================================

Sistema avanzado de análisis de seguridad con detección de amenazas, vulnerabilidades y patrones sospechosos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import re
import hashlib

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Nivel de amenaza."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Tipo de amenaza."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_PAYLOAD = "malicious_payload"


@dataclass
class SecurityThreat:
    """Amenaza de seguridad."""
    threat_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    source: str
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPattern:
    """Patrón de seguridad."""
    pattern_id: str
    pattern_type: ThreatType
    pattern_regex: str
    description: str
    enabled: bool = True


class SecurityAnalyzer:
    """Analizador de seguridad."""
    
    def __init__(self):
        self.threats: List[SecurityThreat] = []
        self.patterns: Dict[str, SecurityPattern] = {}
        self.blocked_sources: Dict[str, datetime] = {}
        self.attack_history: deque = deque(maxlen=100000)
        self._lock = asyncio.Lock()
        
        # Inicializar patrones por defecto
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Inicializar patrones de seguridad por defecto."""
        patterns = [
            SecurityPattern(
                pattern_id="sql_injection_1",
                pattern_type=ThreatType.SQL_INJECTION,
                pattern_regex=r"(?i)(union\s+select|insert\s+into|delete\s+from|drop\s+table|exec\s+\(|script\s*>|--\s*$|/\*|\*/)",
                description="SQL Injection patterns",
            ),
            SecurityPattern(
                pattern_id="xss_1",
                pattern_type=ThreatType.XSS,
                pattern_regex=r"(?i)(<script|javascript:|onerror=|onload=|eval\(|document\.cookie)",
                description="XSS patterns",
            ),
            SecurityPattern(
                pattern_id="path_traversal_1",
                pattern_type=ThreatType.UNAUTHORIZED_ACCESS,
                pattern_regex=r"(\.\./|\.\.\\\\|%2e%2e%2f|%2e%2e%5c)",
                description="Path traversal patterns",
            ),
        ]
        
        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern
    
    def analyze_input(
        self,
        input_data: str,
        source: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SecurityThreat]:
        """Analizar entrada en busca de amenazas."""
        detected_threats = []
        
        for pattern in self.patterns.values():
            if not pattern.enabled:
                continue
            
            matches = re.findall(pattern.pattern_regex, input_data)
            if matches:
                threat_level = self._determine_threat_level(pattern.pattern_type, len(matches))
                
                threat = SecurityThreat(
                    threat_id=f"threat_{source}_{datetime.now().timestamp()}",
                    threat_type=pattern.pattern_type,
                    threat_level=threat_level,
                    source=source,
                    description=f"Detected {pattern.pattern_type.value} pattern: {pattern.description}",
                    metadata={
                        "matches": matches[:5],  # Primeros 5 matches
                        "pattern_id": pattern.pattern_id,
                        "input_length": len(input_data),
                        "context": context or {},
                    },
                )
                
                detected_threats.append(threat)
                
                async def save_threat():
                    async with self._lock:
                        self.threats.append(threat)
                        if len(self.threats) > 10000:
                            self.threats.pop(0)
                        
                        self.attack_history.append({
                            "threat_id": threat.threat_id,
                            "source": source,
                            "timestamp": datetime.now().isoformat(),
                            "threat_type": pattern.pattern_type.value,
                        })
                
                asyncio.create_task(save_threat())
        
        return detected_threats
    
    def _determine_threat_level(self, threat_type: ThreatType, match_count: int) -> ThreatLevel:
        """Determinar nivel de amenaza."""
        critical_types = [ThreatType.SQL_INJECTION, ThreatType.XSS, ThreatType.DATA_EXFILTRATION]
        
        if threat_type in critical_types:
            return ThreatLevel.CRITICAL if match_count > 3 else ThreatLevel.HIGH
        elif match_count > 5:
            return ThreatLevel.HIGH
        elif match_count > 2:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def block_source(self, source: str, duration_seconds: int = 3600):
        """Bloquear fuente temporalmente."""
        async def save_block():
            async with self._lock:
                self.blocked_sources[source] = datetime.now() + timedelta(seconds=duration_seconds)
        
        asyncio.create_task(save_block())
        logger.warning(f"Blocked source: {source} for {duration_seconds} seconds")
    
    def is_source_blocked(self, source: str) -> bool:
        """Verificar si fuente está bloqueada."""
        blocked_until = self.blocked_sources.get(source)
        if not blocked_until:
            return False
        
        if datetime.now() > blocked_until:
            # Limpiar bloqueo expirado
            del self.blocked_sources[source]
            return False
        
        return True
    
    def resolve_threat(self, threat_id: str) -> bool:
        """Resolver amenaza."""
        threat = next((t for t in self.threats if t.threat_id == threat_id), None)
        if not threat:
            return False
        
        threat.resolved = True
        threat.resolved_at = datetime.now()
        return True
    
    def get_threats(
        self,
        threat_type: Optional[ThreatType] = None,
        threat_level: Optional[ThreatLevel] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener amenazas."""
        threats = self.threats
        
        if threat_type:
            threats = [t for t in threats if t.threat_type == threat_type]
        
        if threat_level:
            threats = [t for t in threats if t.threat_level == threat_level]
        
        if resolved is not None:
            threats = [t for t in threats if t.resolved == resolved]
        
        threats.sort(key=lambda t: t.detected_at, reverse=True)
        
        return [
            {
                "threat_id": t.threat_id,
                "threat_type": t.threat_type.value,
                "threat_level": t.threat_level.value,
                "source": t.source,
                "description": t.description,
                "detected_at": t.detected_at.isoformat(),
                "resolved": t.resolved,
                "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
                "metadata": t.metadata,
            }
            for t in threats[:limit]
        ]
    
    def get_security_analyzer_summary(self) -> Dict[str, Any]:
        """Obtener resumen del analizador."""
        by_type: Dict[str, int] = defaultdict(int)
        by_level: Dict[str, int] = defaultdict(int)
        unresolved_count = 0
        
        for threat in self.threats:
            by_type[threat.threat_type.value] += 1
            by_level[threat.threat_level.value] += 1
            if not threat.resolved:
                unresolved_count += 1
        
        return {
            "total_threats": len(self.threats),
            "unresolved_threats": unresolved_count,
            "threats_by_type": dict(by_type),
            "threats_by_level": dict(by_level),
            "blocked_sources": len(self.blocked_sources),
            "total_attacks": len(self.attack_history),
            "active_patterns": len([p for p in self.patterns.values() if p.enabled]),
        }














