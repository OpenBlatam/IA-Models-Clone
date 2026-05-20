"""
Realtime Analyzer - Sistema de análisis de contenido en tiempo real
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RealtimeEvent:
    """Evento en tiempo real"""
    event_type: str
    content_id: str
    data: Dict[str, Any]
    timestamp: datetime


class RealtimeAnalyzer:
    """Analizador en tiempo real"""

    def __init__(self):
        """Inicializar analizador"""
        self.event_queue: deque = deque(maxlen=10000)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.metrics_cache: Dict[str, Dict[str, Any]] = {}
        self.is_running = False

    def subscribe(
        self,
        event_type: str,
        callback: Callable
    ):
        """
        Suscribirse a eventos en tiempo real.

        Args:
            event_type: Tipo de evento
            callback: Función callback
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        logger.debug(f"Suscripción agregada para evento: {event_type}")

    def emit_event(
        self,
        event_type: str,
        content_id: str,
        data: Dict[str, Any]
    ):
        """
        Emitir evento en tiempo real.

        Args:
            event_type: Tipo de evento
            content_id: ID del contenido
            data: Datos del evento
        """
        event = RealtimeEvent(
            event_type=event_type,
            content_id=content_id,
            data=data,
            timestamp=datetime.utcnow()
        )
        
        self.event_queue.append(event)
        
        # Notificar suscriptores
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error en callback de evento {event_type}: {e}")
        
        logger.debug(f"Evento emitido: {event_type} - {content_id}")

    async def analyze_realtime_metrics(
        self,
        content_id: str,
        metric_name: str,
        value: float
    ) -> Dict[str, Any]:
        """
        Analizar métricas en tiempo real.

        Args:
            content_id: ID del contenido
            metric_name: Nombre de la métrica
            value: Valor de la métrica

        Returns:
            Análisis en tiempo real
        """
        cache_key = f"{content_id}:{metric_name}"
        
        if cache_key not in self.metrics_cache:
            self.metrics_cache[cache_key] = {
                "values": deque(maxlen=100),
                "last_update": None
            }
        
        metric_cache = self.metrics_cache[cache_key]
        metric_cache["values"].append(value)
        metric_cache["last_update"] = datetime.utcnow()
        
        # Calcular estadísticas en tiempo real
        values = list(metric_cache["values"])
        if len(values) >= 2:
            current_value = values[-1]
            previous_value = values[-2]
            change = current_value - previous_value
            change_percentage = (change / previous_value * 100) if previous_value != 0 else 0
            
            avg_value = sum(values) / len(values)
            trend = "increasing" if change > 0 else "decreasing" if change < 0 else "stable"
        else:
            current_value = value
            change = 0
            change_percentage = 0
            avg_value = value
            trend = "insufficient_data"
        
        # Emitir evento
        self.emit_event(
            "metric_update",
            content_id,
            {
                "metric_name": metric_name,
                "value": value,
                "change": change,
                "change_percentage": change_percentage,
                "trend": trend
            }
        )
        
        return {
            "content_id": content_id,
            "metric_name": metric_name,
            "current_value": current_value,
            "average_value": avg_value,
            "change": change,
            "change_percentage": change_percentage,
            "trend": trend,
            "data_points": len(values)
        }

    def get_recent_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener eventos recientes.

        Args:
            event_type: Tipo de evento (opcional)
            limit: Límite de eventos

        Returns:
            Lista de eventos
        """
        events = list(self.event_queue)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Ordenar por timestamp (más recientes primero)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "event_type": e.event_type,
                "content_id": e.content_id,
                "data": e.data,
                "timestamp": e.timestamp.isoformat()
            }
            for e in events[:limit]
        ]

    def get_realtime_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas en tiempo real.

        Returns:
            Estadísticas
        """
        total_events = len(self.event_queue)
        
        # Contar eventos por tipo
        event_types = {}
        for event in self.event_queue:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        # Métricas en cache
        active_metrics = len(self.metrics_cache)
        
        return {
            "total_events": total_events,
            "event_types": event_types,
            "active_metrics": active_metrics,
            "subscribers": {
                event_type: len(callbacks)
                for event_type, callbacks in self.subscribers.items()
            }
        }






