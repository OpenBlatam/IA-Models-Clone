"""
Real-time Streaming System
===========================

Sistema de streaming en tiempo real para datos de manufactura.
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
import threading

try:
    from fastapi import WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Tipos de stream."""
    PRODUCTION = "production"
    QUALITY = "quality"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    ALL = "all"


@dataclass
class StreamEvent:
    """Evento de stream."""
    event_type: str
    data: Dict[str, Any]
    timestamp: float
    source: str


class StreamManager:
    """Gestor de streams en tiempo real."""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size
        self.subscribers: Dict[str, Set[Any]] = {}
        self.event_buffer: Dict[str, deque] = {}
        self.lock = threading.RLock()
        self.running = False
    
    def subscribe(self, stream_type: str, subscriber: Any) -> None:
        """Suscribe cliente a stream."""
        with self.lock:
            if stream_type not in self.subscribers:
                self.subscribers[stream_type] = set()
                self.event_buffer[stream_type] = deque(maxlen=self.max_buffer_size)
            self.subscribers[stream_type].add(subscriber)
    
    def unsubscribe(self, stream_type: str, subscriber: Any) -> None:
        """Desuscribe cliente de stream."""
        with self.lock:
            if stream_type in self.subscribers:
                self.subscribers[stream_type].discard(subscriber)
    
    def publish(self, stream_type: str, event: StreamEvent) -> None:
        """Publica evento a suscriptores."""
        with self.lock:
            # Guardar en buffer
            if stream_type in self.event_buffer:
                self.event_buffer[stream_type].append(event)
            
            # Enviar a suscriptores
            if stream_type in self.subscribers:
                disconnected = set()
                for subscriber in self.subscribers[stream_type]:
                    try:
                        self._send_event(subscriber, event)
                    except Exception as e:
                        logger.warning(f"Error sending to subscriber: {e}")
                        disconnected.add(subscriber)
                
                # Limpiar desconectados
                for sub in disconnected:
                    self.subscribers[stream_type].discard(sub)
    
    def _send_event(self, subscriber: Any, event: StreamEvent) -> None:
        """Envía evento a suscriptor."""
        if FASTAPI_AVAILABLE and isinstance(subscriber, WebSocket):
            asyncio.create_task(subscriber.send_json({
                "type": event.event_type,
                "data": event.data,
                "timestamp": event.timestamp,
                "source": event.source
            }))
        elif hasattr(subscriber, 'send'):
            subscriber.send(json.dumps({
                "type": event.event_type,
                "data": event.data,
                "timestamp": event.timestamp,
                "source": event.source
            }))
        elif callable(subscriber):
            subscriber(event)
    
    def get_recent_events(self, stream_type: str, limit: int = 100) -> List[StreamEvent]:
        """Obtiene eventos recientes."""
        with self.lock:
            if stream_type in self.event_buffer:
                return list(self.event_buffer[stream_type])[-limit:]
            return []


class WebSocketStreamHandler:
    """Manejador de WebSocket para streaming."""
    
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, stream_types: List[str]) -> None:
        """Conecta WebSocket."""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        for stream_type in stream_types:
            self.stream_manager.subscribe(stream_type, websocket)
    
    async def disconnect(self, websocket: WebSocket, stream_types: List[str]) -> None:
        """Desconecta WebSocket."""
        self.active_connections.discard(websocket)
        
        for stream_type in stream_types:
            self.stream_manager.unsubscribe(stream_type, websocket)
    
    async def handle_stream(self, websocket: WebSocket, stream_types: List[str]) -> None:
        """Maneja stream de WebSocket."""
        await self.connect(websocket, stream_types)
        
        try:
            while True:
                # Mantener conexión viva
                await websocket.receive_text()
        except WebSocketDisconnect:
            await self.disconnect(websocket, stream_types)


class ProductionStreamer:
    """Streamer para datos de producción."""
    
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
    
    def stream_order_update(self, order_id: str, status: str, data: Dict[str, Any]) -> None:
        """Stream actualización de orden."""
        event = StreamEvent(
            event_type="order_update",
            data={
                "order_id": order_id,
                "status": status,
                **data
            },
            timestamp=time.time(),
            source="production_planner"
        )
        self.stream_manager.publish(StreamType.PRODUCTION.value, event)
    
    def stream_production_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None) -> None:
        """Stream métrica de producción."""
        event = StreamEvent(
            event_type="production_metric",
            data={
                "metric": metric_name,
                "value": value,
                **(metadata or {})
            },
            timestamp=time.time(),
            source="production_monitor"
        )
        self.stream_manager.publish(StreamType.PRODUCTION.value, event)


class QualityStreamer:
    """Streamer para datos de calidad."""
    
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
    
    def stream_inspection_result(self, inspection_id: str, result: Dict[str, Any]) -> None:
        """Stream resultado de inspección."""
        event = StreamEvent(
            event_type="inspection_result",
            data={
                "inspection_id": inspection_id,
                **result
            },
            timestamp=time.time(),
            source="quality_control"
        )
        self.stream_manager.publish(StreamType.QUALITY.value, event)
    
    def stream_defect_detected(self, defect_type: str, severity: str, location: Dict[str, Any]) -> None:
        """Stream defecto detectado."""
        event = StreamEvent(
            event_type="defect_detected",
            data={
                "defect_type": defect_type,
                "severity": severity,
                "location": location
            },
            timestamp=time.time(),
            source="quality_control"
        )
        self.stream_manager.publish(StreamType.QUALITY.value, event)


class MonitoringStreamer:
    """Streamer para monitoreo de equipos."""
    
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
    
    def stream_equipment_status(self, equipment_id: str, status: str, metrics: Dict[str, Any]) -> None:
        """Stream estado de equipo."""
        event = StreamEvent(
            event_type="equipment_status",
            data={
                "equipment_id": equipment_id,
                "status": status,
                "metrics": metrics
            },
            timestamp=time.time(),
            source="monitoring"
        )
        self.stream_manager.publish(StreamType.MONITORING.value, event)
    
    def stream_alert(self, alert_type: str, message: str, severity: str, data: Optional[Dict] = None) -> None:
        """Stream alerta."""
        event = StreamEvent(
            event_type="alert",
            data={
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                **(data or {})
            },
            timestamp=time.time(),
            source="monitoring"
        )
        self.stream_manager.publish(StreamType.MONITORING.value, event)


class OptimizationStreamer:
    """Streamer para optimizaciones."""
    
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
    
    def stream_optimization_result(self, optimization_id: str, result: Dict[str, Any]) -> None:
        """Stream resultado de optimización."""
        event = StreamEvent(
            event_type="optimization_result",
            data={
                "optimization_id": optimization_id,
                **result
            },
            timestamp=time.time(),
            source="process_optimizer"
        )
        self.stream_manager.publish(StreamType.OPTIMIZATION.value, event)
    
    def stream_recommendation(self, recommendation_type: str, recommendation: Dict[str, Any]) -> None:
        """Stream recomendación."""
        event = StreamEvent(
            event_type="recommendation",
            data={
                "type": recommendation_type,
                **recommendation
            },
            timestamp=time.time(),
            source="optimizer"
        )
        self.stream_manager.publish(StreamType.OPTIMIZATION.value, event)


class StreamAggregator:
    """Agrega múltiples streams."""
    
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.aggregation_window = 1.0  # segundos
        self.aggregated_data: Dict[str, deque] = {}
    
    def aggregate(self, stream_type: str, window_size: int = 10) -> Dict[str, Any]:
        """Agrega eventos en ventana temporal."""
        events = self.stream_manager.get_recent_events(stream_type, limit=window_size)
        
        if not events:
            return {}
        
        # Agregar por tipo de evento
        aggregated = {}
        for event in events:
            event_type = event.event_type
            if event_type not in aggregated:
                aggregated[event_type] = {
                    "count": 0,
                    "latest": None,
                    "values": []
                }
            
            aggregated[event_type]["count"] += 1
            aggregated[event_type]["latest"] = event.data
            aggregated[event_type]["values"].append(event.data)
        
        return aggregated


# Factory functions
_stream_manager = None

def get_stream_manager() -> StreamManager:
    """Obtiene instancia global de StreamManager."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager


