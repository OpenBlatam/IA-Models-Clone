"""
MCP Analytics - Analytics y reporting
=======================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from collections import defaultdict

logger = logging.getLogger(__name__)


class AnalyticsEvent(BaseModel):
    """Evento de analytics"""
    event_id: str = Field(..., description="ID único del evento")
    event_type: str = Field(..., description="Tipo de evento")
    resource_id: Optional[str] = Field(None, description="ID del recurso")
    user_id: Optional[str] = Field(None, description="ID del usuario")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalyticsCollector:
    """
    Recolector de analytics
    
    Recolecta y analiza métricas de uso del sistema.
    """
    
    def __init__(self):
        self._events: List[AnalyticsEvent] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, List[float]] = defaultdict(list)
    
    def record_event(
        self,
        event_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Registra un evento
        
        Args:
            event_type: Tipo de evento
            resource_id: ID del recurso (opcional)
            user_id: ID del usuario (opcional)
            metadata: Metadata adicional (opcional)
        """
        import uuid
        
        event = AnalyticsEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            resource_id=resource_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        
        self._events.append(event)
        self._counters[event_type] += 1
        
        # Mantener solo últimos 10000 eventos
        if len(self._events) > 10000:
            self._events = self._events[-10000:]
    
    def record_timing(self, operation: str, duration: float):
        """
        Registra tiempo de operación
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
        """
        self._timers[operation].append(duration)
        
        # Mantener solo últimos 1000 timings por operación
        if len(self._timers[operation]) > 1000:
            self._timers[operation] = self._timers[operation][-1000:]
    
    def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas
        
        Args:
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
            
        Returns:
            Diccionario con estadísticas
        """
        events = self._events
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Estadísticas de eventos
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event.event_type] += 1
        
        # Estadísticas de timings
        timing_stats = {}
        for operation, timings in self._timers.items():
            if timings:
                timing_stats[operation] = {
                    "count": len(timings),
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                }
        
        return {
            "total_events": len(events),
            "event_counts": dict(event_counts),
            "timing_stats": timing_stats,
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        }
    
    def get_resource_stats(
        self,
        resource_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un recurso
        
        Args:
            resource_id: ID del recurso
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
            
        Returns:
            Diccionario con estadísticas del recurso
        """
        events = [
            e for e in self._events
            if e.resource_id == resource_id
        ]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event.event_type] += 1
        
        return {
            "resource_id": resource_id,
            "total_events": len(events),
            "event_counts": dict(event_counts),
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        }
    
    def clear(self):
        """Limpia todos los datos de analytics"""
        self._events.clear()
        self._counters.clear()
        self._timers.clear()
        logger.info("Analytics data cleared")

