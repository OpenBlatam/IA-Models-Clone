"""
Query Analyzer - Analizador de Queries
======================================

Sistema de análisis de performance de queries con detección de queries lentas y optimización.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import re

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Tipo de query."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"


@dataclass
class QueryExecution:
    """Ejecución de query."""
    query_id: str
    query_text: str
    query_type: QueryType
    execution_time: float
    timestamp: datetime
    rows_affected: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPattern:
    """Patrón de query."""
    pattern_id: str
    query_pattern: str  # Patrón normalizado
    query_type: QueryType
    total_executions: int = 0
    avg_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0
    error_count: int = 0
    last_executed: Optional[datetime] = None


class QueryAnalyzer:
    """Analizador de queries."""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.query_history: deque = deque(maxlen=history_size)
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.slow_queries: List[str] = []  # query_ids
        self._lock = asyncio.Lock()
        self.slow_query_threshold: float = 1.0  # segundos
    
    def normalize_query(self, query_text: str) -> str:
        """Normalizar query para agrupar similares."""
        # Remover espacios extra
        normalized = re.sub(r'\s+', ' ', query_text.strip())
        
        # Reemplazar valores literales con placeholders
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'"[^"]*"', '"?"', normalized)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        
        # Normalizar a minúsculas
        normalized = normalized.lower()
        
        return normalized
    
    def detect_query_type(self, query_text: str) -> QueryType:
        """Detectar tipo de query."""
        query_lower = query_text.strip().lower()
        
        if query_lower.startswith('select'):
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        elif query_lower.startswith('create'):
            return QueryType.CREATE
        elif query_lower.startswith('drop'):
            return QueryType.DROP
        elif query_lower.startswith('alter'):
            return QueryType.ALTER
        else:
            return QueryType.SELECT  # Default
    
    def record_query(
        self,
        query_text: str,
        execution_time: float,
        rows_affected: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar ejecución de query."""
        query_id = f"query_{datetime.now().timestamp()}"
        
        query_type = self.detect_query_type(query_text)
        
        execution = QueryExecution(
            query_id=query_id,
            query_text=query_text,
            query_type=query_type,
            execution_time=execution_time,
            timestamp=datetime.now(),
            rows_affected=rows_affected,
            error=error,
            metadata=metadata or {},
        )
        
        async def process_query():
            async with self._lock:
                self.query_history.append(execution)
                
                # Detectar query lenta
                if execution_time > self.slow_query_threshold:
                    self.slow_queries.append(query_id)
                    if len(self.slow_queries) > 1000:
                        self.slow_queries.pop(0)
                
                # Actualizar patrones
                pattern_text = self.normalize_query(query_text)
                pattern_id = hashlib.md5(pattern_text.encode()).hexdigest()
                
                if pattern_id not in self.query_patterns:
                    pattern = QueryPattern(
                        pattern_id=pattern_id,
                        query_pattern=pattern_text,
                        query_type=query_type,
                    )
                    self.query_patterns[pattern_id] = pattern
                
                pattern = self.query_patterns[pattern_id]
                pattern.total_executions += 1
                pattern.last_executed = datetime.now()
                
                if error:
                    pattern.error_count += 1
        
        asyncio.create_task(process_query())
        
        # Actualizar estadísticas de patrones
        asyncio.create_task(self._update_pattern_stats(pattern_id))
        
        return query_id
    
    async def _update_pattern_stats(self, pattern_id: str):
        """Actualizar estadísticas de patrón."""
        pattern = self.query_patterns.get(pattern_id)
        if not pattern:
            return
        
        # Obtener ejecuciones recientes de este patrón
        pattern_text = pattern.query_pattern
        recent_executions = [
            e for e in self.query_history
            if self.normalize_query(e.query_text) == pattern_text
        ][-100:]  # Últimas 100
        
        if not recent_executions:
            return
        
        execution_times = [e.execution_time for e in recent_executions]
        execution_times.sort()
        
        async with self._lock:
            pattern.avg_execution_time = statistics.mean(execution_times)
            
            if len(execution_times) > 0:
                pattern.p95_execution_time = execution_times[int(len(execution_times) * 0.95)] if len(execution_times) > 20 else execution_times[-1]
                pattern.p99_execution_time = execution_times[int(len(execution_times) * 0.99)] if len(execution_times) > 100 else execution_times[-1]
    
    def get_slow_queries(
        self,
        threshold: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Obtener queries lentas."""
        threshold = threshold or self.slow_query_threshold
        
        slow = [
            {
                "query_id": e.query_id,
                "query_text": e.query_text[:200],  # Truncar
                "query_type": e.query_type.value,
                "execution_time": e.execution_time,
                "timestamp": e.timestamp.isoformat(),
                "error": e.error,
            }
            for e in self.query_history
            if e.execution_time > threshold
        ]
        
        slow.sort(key=lambda x: x["execution_time"], reverse=True)
        return slow[:limit]
    
    def get_query_patterns(
        self,
        query_type: Optional[QueryType] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Obtener patrones de queries."""
        patterns = list(self.query_patterns.values())
        
        if query_type:
            patterns = [p for p in patterns if p.query_type == query_type]
        
        patterns.sort(key=lambda p: p.avg_execution_time, reverse=True)
        
        return [
            {
                "pattern_id": p.pattern_id,
                "query_pattern": p.query_pattern[:200],  # Truncar
                "query_type": p.query_type.value,
                "total_executions": p.total_executions,
                "avg_execution_time": p.avg_execution_time,
                "p95_execution_time": p.p95_execution_time,
                "p99_execution_time": p.p99_execution_time,
                "error_count": p.error_count,
                "error_rate": p.error_count / p.total_executions if p.total_executions > 0 else 0.0,
                "last_executed": p.last_executed.isoformat() if p.last_executed else None,
            }
            for p in patterns[:limit]
        ]
    
    def get_query_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Obtener estadísticas de queries."""
        queries = list(self.query_history)
        
        if start_time:
            queries = [q for q in queries if q.timestamp >= start_time]
        
        if end_time:
            queries = [q for q in queries if q.timestamp <= end_time]
        
        if not queries:
            return {}
        
        execution_times = [q.execution_time for q in queries]
        by_type: Dict[str, int] = defaultdict(int)
        error_count = 0
        
        for q in queries:
            by_type[q.query_type.value] += 1
            if q.error:
                error_count += 1
        
        return {
            "total_queries": len(queries),
            "queries_by_type": dict(by_type),
            "avg_execution_time": statistics.mean(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "p95_execution_time": sorted(execution_times)[int(len(execution_times) * 0.95)] if len(execution_times) > 20 else max(execution_times),
            "p99_execution_time": sorted(execution_times)[int(len(execution_times) * 0.99)] if len(execution_times) > 100 else max(execution_times),
            "error_count": error_count,
            "error_rate": error_count / len(queries) if queries else 0.0,
            "slow_queries_count": len([q for q in queries if q.execution_time > self.slow_query_threshold]),
        }
    
    def get_query_analyzer_summary(self) -> Dict[str, Any]:
        """Obtener resumen del analizador."""
        return {
            "total_queries": len(self.query_history),
            "total_patterns": len(self.query_patterns),
            "slow_queries_count": len(self.slow_queries),
            "slow_query_threshold": self.slow_query_threshold,
        }
















