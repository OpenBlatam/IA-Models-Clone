"""
Metrics and Monitoring
======================

Sistema de métricas y monitoreo para el chat continuo.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class ChatMetrics:
    """Métricas de una sesión de chat."""
    session_id: str
    messages_sent: int = 0
    messages_received: int = 0
    total_responses: int = 0
    total_tokens_generated: int = 0
    total_tokens_prompt: int = 0
    average_response_time: float = 0.0
    total_pause_time: float = 0.0
    pause_count: int = 0
    error_count: int = 0
    last_activity: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    start_time: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Recolector de métricas."""
    
    def __init__(self):
        self.metrics: Dict[str, ChatMetrics] = {}
        self.global_metrics = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_messages": 0,
            "total_errors": 0,
            "average_response_time": 0.0,
        }
        self._lock = asyncio.Lock()
    
    def get_or_create_metrics(self, session_id: str) -> ChatMetrics:
        """Obtener o crear métricas para una sesión."""
        if session_id not in self.metrics:
            self.metrics[session_id] = ChatMetrics(session_id=session_id)
        return self.metrics[session_id]
    
    async def record_message_sent(self, session_id: str):
        """Registrar mensaje enviado."""
        async with self._lock:
            metrics = self.get_or_create_metrics(session_id)
            metrics.messages_sent += 1
            metrics.last_activity = datetime.now()
            self.global_metrics["total_messages"] += 1
    
    async def record_message_received(self, session_id: str):
        """Registrar mensaje recibido."""
        async with self._lock:
            metrics = self.get_or_create_metrics(session_id)
            metrics.messages_received += 1
            metrics.last_activity = datetime.now()
    
    async def record_response(
        self,
        session_id: str,
        response_time: float,
        tokens_generated: int = 0,
        tokens_prompt: int = 0,
    ):
        """Registrar respuesta generada."""
        async with self._lock:
            metrics = self.get_or_create_metrics(session_id)
            metrics.total_responses += 1
            metrics.total_tokens_generated += tokens_generated
            metrics.total_tokens_prompt += tokens_prompt
            metrics.response_times.append(response_time)
            
            # Calcular promedio de tiempo de respuesta
            if metrics.response_times:
                metrics.average_response_time = sum(metrics.response_times) / len(metrics.response_times)
            
            # Actualizar métricas globales
            all_response_times = []
            for m in self.metrics.values():
                all_response_times.extend(m.response_times)
            
            if all_response_times:
                self.global_metrics["average_response_time"] = sum(all_response_times) / len(all_response_times)
    
    async def record_error(self, session_id: str, error_type: str = "unknown"):
        """Registrar error."""
        async with self._lock:
            metrics = self.get_or_create_metrics(session_id)
            metrics.error_count += 1
            self.global_metrics["total_errors"] += 1
    
    async def record_pause(self, session_id: str, pause_duration: float):
        """Registrar pausa."""
        async with self._lock:
            metrics = self.get_or_create_metrics(session_id)
            metrics.pause_count += 1
            metrics.total_pause_time += pause_duration
    
    async def record_session_created(self, session_id: str):
        """Registrar creación de sesión."""
        async with self._lock:
            self.global_metrics["total_sessions"] += 1
            self.global_metrics["active_sessions"] += 1
    
    async def record_session_ended(self, session_id: str):
        """Registrar fin de sesión."""
        async with self._lock:
            self.global_metrics["active_sessions"] = max(0, self.global_metrics["active_sessions"] - 1)
            if session_id in self.metrics:
                del self.metrics[session_id]
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtener métricas de una sesión."""
        if session_id not in self.metrics:
            return None
        
        metrics = self.metrics[session_id]
        uptime = (datetime.now() - metrics.start_time).total_seconds()
        
        return {
            "session_id": session_id,
            "messages_sent": metrics.messages_sent,
            "messages_received": metrics.messages_received,
            "total_responses": metrics.total_responses,
            "total_tokens_generated": metrics.total_tokens_generated,
            "total_tokens_prompt": metrics.total_tokens_prompt,
            "average_response_time": metrics.average_response_time,
            "total_pause_time": metrics.total_pause_time,
            "pause_count": metrics.pause_count,
            "error_count": metrics.error_count,
            "uptime_seconds": uptime,
            "last_activity": metrics.last_activity.isoformat() if metrics.last_activity else None,
        }
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Obtener métricas globales."""
        return {
            **self.global_metrics,
            "active_sessions": len(self.metrics),
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas."""
        return {
            "global": self.get_global_metrics(),
            "sessions": {
                session_id: self.get_session_metrics(session_id)
                for session_id in self.metrics.keys()
            },
        }
































