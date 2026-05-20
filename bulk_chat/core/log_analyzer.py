"""
Log Analyzer - Analizador de Logs
==================================

Sistema avanzado de análisis de logs con búsqueda, filtrado, agregación y detección de patrones.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Nivel de log."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogPattern(Enum):
    """Patrón de log."""
    ERROR_SPIKE = "error_spike"
    SLOW_RESPONSE = "slow_response"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT = "rate_limit"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"


@dataclass
class LogEntry:
    """Entrada de log."""
    entry_id: str
    timestamp: datetime
    level: LogLevel
    message: str
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogPatternMatch:
    """Coincidencia de patrón."""
    pattern: LogPattern
    entry_id: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


class LogAnalyzer:
    """Analizador de logs."""
    
    def __init__(self, max_entries: int = 100000):
        self.max_entries = max_entries
        self.logs: List[LogEntry] = []
        self.index_by_level: Dict[LogLevel, List[str]] = defaultdict(list)
        self.index_by_source: Dict[str, List[str]] = defaultdict(list)
        self.pattern_matches: List[LogPatternMatch] = []
        self._lock = asyncio.Lock()
    
    def ingest_log(
        self,
        level: LogLevel,
        message: str,
        source: str = "",
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ingerir log."""
        entry_id = f"log_{datetime.now().timestamp()}"
        
        entry = LogEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source,
            context=context or {},
            metadata=metadata or {},
        )
        
        self.logs.append(entry)
        self.index_by_level[level].append(entry_id)
        self.index_by_source[source].append(entry_id)
        
        # Limitar tamaño
        if len(self.logs) > self.max_entries:
            removed = self.logs.pop(0)
            self.index_by_level[removed.level].remove(removed.entry_id)
            self.index_by_source[removed.source].remove(removed.entry_id)
        
        # Detectar patrones automáticamente
        asyncio.create_task(self._detect_patterns(entry))
        
        return entry_id
    
    async def _detect_patterns(self, entry: LogEntry):
        """Detectar patrones en log."""
        # Detectar error spike
        if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            recent_errors = [
                e for e in self.logs[-100:]
                if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]
                and (datetime.now() - e.timestamp).total_seconds() < 60
            ]
            
            if len(recent_errors) > 10:
                match = LogPatternMatch(
                    pattern=LogPattern.ERROR_SPIKE,
                    entry_id=entry.entry_id,
                    confidence=min(1.0, len(recent_errors) / 20.0),
                    details={"error_count": len(recent_errors)},
                )
                
                async with self._lock:
                    self.pattern_matches.append(match)
        
        # Detectar auth failures
        if "auth" in entry.message.lower() and "fail" in entry.message.lower():
            match = LogPatternMatch(
                pattern=LogPattern.AUTH_FAILURE,
                entry_id=entry.entry_id,
                confidence=0.8,
                details={"source": entry.source},
            )
            
            async with self._lock:
                self.pattern_matches.append(match)
    
    async def search_logs(
        self,
        query: str,
        level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Buscar logs."""
        candidates = self.logs
        
        # Filtrar por nivel
        if level:
            candidate_ids = set(self.index_by_level.get(level, []))
            candidates = [e for e in candidates if e.entry_id in candidate_ids]
        
        # Filtrar por fuente
        if source:
            candidate_ids = set(self.index_by_source.get(source, []))
            candidates = [e for e in candidates if e.entry_id in candidate_ids]
        
        # Filtrar por tiempo
        if start_time:
            candidates = [e for e in candidates if e.timestamp >= start_time]
        
        if end_time:
            candidates = [e for e in candidates if e.timestamp <= end_time]
        
        # Buscar por query (regex o texto)
        query_lower = query.lower()
        matches = []
        
        for entry in candidates:
            if query_lower in entry.message.lower():
                matches.append(entry)
        
        # Ordenar por timestamp (más recientes primero)
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        
        return [
            {
                "entry_id": e.entry_id,
                "timestamp": e.timestamp.isoformat(),
                "level": e.level.value,
                "message": e.message,
                "source": e.source,
                "context": e.context,
            }
            for e in matches[:limit]
        ]
    
    def get_log_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Obtener estadísticas de logs."""
        logs = self.logs
        
        if start_time:
            logs = [e for e in logs if e.timestamp >= start_time]
        
        if end_time:
            logs = [e for e in logs if e.timestamp <= end_time]
        
        by_level: Dict[str, int] = defaultdict(int)
        by_source: Dict[str, int] = defaultdict(int)
        
        for entry in logs:
            by_level[entry.level.value] += 1
            by_source[entry.source] += 1
        
        return {
            "total_logs": len(logs),
            "logs_by_level": dict(by_level),
            "logs_by_source": dict(by_source),
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        }
    
    def get_pattern_matches(
        self,
        pattern: Optional[LogPattern] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener coincidencias de patrones."""
        matches = self.pattern_matches
        
        if pattern:
            matches = [m for m in matches if m.pattern == pattern]
        
        return [
            {
                "pattern": m.pattern.value,
                "entry_id": m.entry_id,
                "confidence": m.confidence,
                "details": m.details,
            }
            for m in matches[-limit:]
        ]
    
    def get_log_analysis_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis."""
        by_pattern: Dict[str, int] = defaultdict(int)
        
        for match in self.pattern_matches:
            by_pattern[match.pattern.value] += 1
        
        return {
            "total_logs": len(self.logs),
            "total_pattern_matches": len(self.pattern_matches),
            "matches_by_pattern": dict(by_pattern),
        }
















