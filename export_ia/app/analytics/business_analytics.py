"""
Business Analytics - Sistema de análisis de negocio avanzado
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)


class AnalyticsEventType(Enum):
    """Tipos de eventos de analytics."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    BUSINESS_METRIC = "business_metric"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    SECURITY_EVENT = "security_event"


class MetricType(Enum):
    """Tipos de métricas."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    CUSTOM = "custom"


@dataclass
class AnalyticsEvent:
    """Evento de analytics."""
    event_id: str
    event_type: AnalyticsEventType
    name: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessMetric:
    """Métrica de negocio."""
    metric_id: str
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """Sesión de usuario."""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[AnalyticsEvent] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class BusinessAnalytics:
    """
    Sistema de análisis de negocio avanzado.
    """
    
    def __init__(self, retention_days: int = 90):
        """Inicializar sistema de analytics."""
        self.retention_days = retention_days
        
        # Almacenamiento de datos
        self.events: List[AnalyticsEvent] = []
        self.metrics: List[BusinessMetric] = []
        self.user_sessions: Dict[str, UserSession] = {}
        
        # Índices para búsqueda rápida
        self.events_by_type: Dict[AnalyticsEventType, List[AnalyticsEvent]] = defaultdict(list)
        self.events_by_user: Dict[str, List[AnalyticsEvent]] = defaultdict(list)
        self.events_by_session: Dict[str, List[AnalyticsEvent]] = defaultdict(list)
        self.metrics_by_name: Dict[str, List[BusinessMetric]] = defaultdict(list)
        
        # Configuración
        self.max_events = 1000000  # 1M eventos
        self.max_metrics = 100000   # 100K métricas
        self.cleanup_interval = 3600  # 1 hora
        
        # Estadísticas
        self.stats = {
            "total_events": 0,
            "total_metrics": 0,
            "total_sessions": 0,
            "active_sessions": 0,
            "start_time": datetime.now()
        }
        
        # Iniciar tareas de limpieza
        self._cleanup_task = None
        
        logger.info("BusinessAnalytics inicializado")
    
    async def initialize(self):
        """Inicializar el sistema de analytics."""
        try:
            # Iniciar tarea de limpieza
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("BusinessAnalytics inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar BusinessAnalytics: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el sistema de analytics."""
        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("BusinessAnalytics cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar BusinessAnalytics: {e}")
    
    async def track_event(
        self,
        event_type: AnalyticsEventType,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Rastrear un evento."""
        try:
            event_id = str(uuid.uuid4())
            now = datetime.now()
            
            event = AnalyticsEvent(
                event_id=event_id,
                event_type=event_type,
                name=name,
                timestamp=now,
                user_id=user_id,
                session_id=session_id,
                properties=properties or {},
                metrics=metrics or {},
                tags=tags or {},
                metadata=metadata or {}
            )
            
            # Almacenar evento
            self.events.append(event)
            self.stats["total_events"] += 1
            
            # Actualizar índices
            self.events_by_type[event_type].append(event)
            if user_id:
                self.events_by_user[user_id].append(event)
            if session_id:
                self.events_by_session[session_id].append(event)
            
            # Actualizar sesión de usuario
            if session_id:
                await self._update_user_session(session_id, event, user_id)
            
            # Limpiar eventos antiguos si es necesario
            if len(self.events) > self.max_events:
                await self._cleanup_old_events()
            
            logger.debug(f"Evento rastreado: {name} ({event_type.value})")
            return event_id
            
        except Exception as e:
            logger.error(f"Error al rastrear evento: {e}")
            raise
    
    async def track_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        dimensions: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Rastrear una métrica."""
        try:
            metric_id = str(uuid.uuid4())
            now = datetime.now()
            
            metric = BusinessMetric(
                metric_id=metric_id,
                name=name,
                metric_type=metric_type,
                value=value,
                timestamp=now,
                dimensions=dimensions or {},
                metadata=metadata or {}
            )
            
            # Almacenar métrica
            self.metrics.append(metric)
            self.stats["total_metrics"] += 1
            
            # Actualizar índice
            self.metrics_by_name[name].append(metric)
            
            # Limpiar métricas antiguas si es necesario
            if len(self.metrics) > self.max_metrics:
                await self._cleanup_old_metrics()
            
            logger.debug(f"Métrica rastreada: {name} = {value}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error al rastrear métrica: {e}")
            raise
    
    async def start_user_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        properties: Dict[str, Any] = None
    ) -> str:
        """Iniciar sesión de usuario."""
        try:
            now = datetime.now()
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                start_time=now,
                properties=properties or {}
            )
            
            self.user_sessions[session_id] = session
            self.stats["total_sessions"] += 1
            self.stats["active_sessions"] += 1
            
            # Rastrear evento de inicio de sesión
            await self.track_event(
                event_type=AnalyticsEventType.USER_ACTION,
                name="session_started",
                user_id=user_id,
                session_id=session_id,
                properties={"session_properties": properties or {}}
            )
            
            logger.info(f"Sesión iniciada: {session_id} (usuario: {user_id})")
            return session_id
            
        except Exception as e:
            logger.error(f"Error al iniciar sesión: {e}")
            raise
    
    async def end_user_session(self, session_id: str) -> bool:
        """Finalizar sesión de usuario."""
        try:
            if session_id not in self.user_sessions:
                return False
            
            session = self.user_sessions[session_id]
            session.end_time = datetime.now()
            session.is_active = False
            
            self.stats["active_sessions"] -= 1
            
            # Calcular duración de sesión
            duration = (session.end_time - session.start_time).total_seconds()
            
            # Rastrear evento de fin de sesión
            await self.track_event(
                event_type=AnalyticsEventType.USER_ACTION,
                name="session_ended",
                user_id=session.user_id,
                session_id=session_id,
                properties={"duration_seconds": duration}
            )
            
            # Rastrear métrica de duración de sesión
            await self.track_metric(
                name="session_duration_seconds",
                value=duration,
                metric_type=MetricType.HISTOGRAM,
                dimensions={"user_id": session.user_id or "anonymous"}
            )
            
            logger.info(f"Sesión finalizada: {session_id} (duración: {duration:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error al finalizar sesión: {e}")
            return False
    
    async def _update_user_session(self, session_id: str, event: AnalyticsEvent, user_id: Optional[str]):
        """Actualizar sesión de usuario."""
        try:
            if session_id not in self.user_sessions:
                # Crear sesión automáticamente si no existe
                await self.start_user_session(session_id, user_id)
            
            session = self.user_sessions[session_id]
            session.events.append(event)
            
        except Exception as e:
            logger.error(f"Error al actualizar sesión: {e}")
    
    async def _cleanup_loop(self):
        """Bucle de limpieza automática."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Error en bucle de limpieza: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def _cleanup_old_data(self):
        """Limpiar datos antiguos."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            # Limpiar eventos antiguos
            await self._cleanup_old_events(cutoff_time)
            
            # Limpiar métricas antiguas
            await self._cleanup_old_metrics(cutoff_time)
            
            # Limpiar sesiones antiguas
            await self._cleanup_old_sessions(cutoff_time)
            
        except Exception as e:
            logger.error(f"Error en limpieza de datos: {e}")
    
    async def _cleanup_old_events(self, cutoff_time: Optional[datetime] = None):
        """Limpiar eventos antiguos."""
        try:
            if cutoff_time is None:
                # Mantener solo los eventos más recientes
                if len(self.events) > self.max_events:
                    # Eliminar los eventos más antiguos
                    events_to_remove = len(self.events) - self.max_events
                    self.events = self.events[events_to_remove:]
                    
                    # Reconstruir índices
                    self._rebuild_event_indexes()
            else:
                # Eliminar eventos más antiguos que cutoff_time
                old_events = [e for e in self.events if e.timestamp < cutoff_time]
                self.events = [e for e in self.events if e.timestamp >= cutoff_time]
                
                # Actualizar índices
                for event in old_events:
                    self.events_by_type[event.event_type].remove(event)
                    if event.user_id:
                        self.events_by_user[event.user_id].remove(event)
                    if event.session_id:
                        self.events_by_session[event.session_id].remove(event)
            
        except Exception as e:
            logger.error(f"Error al limpiar eventos antiguos: {e}")
    
    async def _cleanup_old_metrics(self, cutoff_time: Optional[datetime] = None):
        """Limpiar métricas antiguas."""
        try:
            if cutoff_time is None:
                # Mantener solo las métricas más recientes
                if len(self.metrics) > self.max_metrics:
                    # Eliminar las métricas más antiguas
                    metrics_to_remove = len(self.metrics) - self.max_metrics
                    self.metrics = self.metrics[metrics_to_remove:]
                    
                    # Reconstruir índices
                    self._rebuild_metric_indexes()
            else:
                # Eliminar métricas más antiguas que cutoff_time
                old_metrics = [m for m in self.metrics if m.timestamp < cutoff_time]
                self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
                
                # Actualizar índices
                for metric in old_metrics:
                    self.metrics_by_name[metric.name].remove(metric)
            
        except Exception as e:
            logger.error(f"Error al limpiar métricas antiguas: {e}")
    
    async def _cleanup_old_sessions(self, cutoff_time: datetime):
        """Limpiar sesiones antiguas."""
        try:
            sessions_to_remove = []
            for session_id, session in self.user_sessions.items():
                if session.start_time < cutoff_time and not session.is_active:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.user_sessions[session_id]
            
        except Exception as e:
            logger.error(f"Error al limpiar sesiones antiguas: {e}")
    
    def _rebuild_event_indexes(self):
        """Reconstruir índices de eventos."""
        try:
            self.events_by_type.clear()
            self.events_by_user.clear()
            self.events_by_session.clear()
            
            for event in self.events:
                self.events_by_type[event.event_type].append(event)
                if event.user_id:
                    self.events_by_user[event.user_id].append(event)
                if event.session_id:
                    self.events_by_session[event.session_id].append(event)
            
        except Exception as e:
            logger.error(f"Error al reconstruir índices de eventos: {e}")
    
    def _rebuild_metric_indexes(self):
        """Reconstruir índices de métricas."""
        try:
            self.metrics_by_name.clear()
            
            for metric in self.metrics:
                self.metrics_by_name[metric.name].append(metric)
            
        except Exception as e:
            logger.error(f"Error al reconstruir índices de métricas: {e}")
    
    async def get_events(
        self,
        event_type: Optional[AnalyticsEventType] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener eventos."""
        try:
            events = self.events.copy()
            
            # Filtrar por tipo
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Filtrar por usuario
            if user_id:
                events = [e for e in events if e.user_id == user_id]
            
            # Filtrar por sesión
            if session_id:
                events = [e for e in events if e.session_id == session_id]
            
            # Filtrar por tiempo
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            
            # Ordenar por timestamp
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            events = events[:limit]
            
            # Convertir a diccionario
            return [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "user_id": e.user_id,
                    "session_id": e.session_id,
                    "properties": e.properties,
                    "metrics": e.metrics,
                    "tags": e.tags,
                    "metadata": e.metadata
                }
                for e in events
            ]
            
        except Exception as e:
            logger.error(f"Error al obtener eventos: {e}")
            return []
    
    async def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener métricas."""
        try:
            metrics = self.metrics.copy()
            
            # Filtrar por nombre
            if metric_name:
                metrics = [m for m in metrics if m.name == metric_name]
            
            # Filtrar por tiempo
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            # Ordenar por timestamp
            metrics.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            metrics = metrics[:limit]
            
            # Convertir a diccionario
            return [
                {
                    "metric_id": m.metric_id,
                    "name": m.name,
                    "metric_type": m.metric_type.value,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "dimensions": m.dimensions,
                    "metadata": m.metadata
                }
                for m in metrics
            ]
            
        except Exception as e:
            logger.error(f"Error al obtener métricas: {e}")
            return []
    
    async def get_analytics_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Obtener resumen de analytics."""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(days=30)
            if end_time is None:
                end_time = datetime.now()
            
            # Filtrar eventos por tiempo
            filtered_events = [
                e for e in self.events
                if start_time <= e.timestamp <= end_time
            ]
            
            # Filtrar métricas por tiempo
            filtered_metrics = [
                m for m in self.metrics
                if start_time <= m.timestamp <= end_time
            ]
            
            # Estadísticas de eventos
            event_counts = Counter(e.name for e in filtered_events)
            event_types = Counter(e.event_type.value for e in filtered_events)
            
            # Estadísticas de usuarios
            unique_users = len(set(e.user_id for e in filtered_events if e.user_id))
            unique_sessions = len(set(e.session_id for e in filtered_events if e.session_id))
            
            # Estadísticas de métricas
            metric_names = Counter(m.name for m in filtered_metrics)
            
            # Calcular estadísticas de métricas numéricas
            metric_stats = {}
            for metric_name in metric_names.keys():
                values = [m.value for m in filtered_metrics if m.name == metric_name]
                if values:
                    metric_stats[metric_name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": statistics.mean(values),
                        "median": statistics.median(values)
                    }
            
            return {
                "period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_days": (end_time - start_time).days
                },
                "events": {
                    "total": len(filtered_events),
                    "by_name": dict(event_counts),
                    "by_type": dict(event_types),
                    "unique_users": unique_users,
                    "unique_sessions": unique_sessions
                },
                "metrics": {
                    "total": len(filtered_metrics),
                    "by_name": dict(metric_names),
                    "statistics": metric_stats
                },
                "sessions": {
                    "total": len(self.user_sessions),
                    "active": len([s for s in self.user_sessions.values() if s.is_active])
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener resumen de analytics: {e}")
            return {}
    
    async def get_user_analytics(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Obtener analytics de usuario específico."""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(days=30)
            if end_time is None:
                end_time = datetime.now()
            
            # Obtener eventos del usuario
            user_events = await self.get_events(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            # Obtener sesiones del usuario
            user_sessions = [
                s for s in self.user_sessions.values()
                if s.user_id == user_id and start_time <= s.start_time <= end_time
            ]
            
            # Calcular estadísticas
            event_counts = Counter(e["name"] for e in user_events)
            session_durations = [
                (s.end_time - s.start_time).total_seconds()
                for s in user_sessions if s.end_time
            ]
            
            return {
                "user_id": user_id,
                "period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                },
                "events": {
                    "total": len(user_events),
                    "by_name": dict(event_counts)
                },
                "sessions": {
                    "total": len(user_sessions),
                    "active": len([s for s in user_sessions if s.is_active]),
                    "avg_duration_seconds": statistics.mean(session_durations) if session_durations else 0,
                    "total_duration_seconds": sum(session_durations)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener analytics de usuario: {e}")
            return {}
    
    async def get_analytics_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema de analytics."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "retention_days": self.retention_days,
            "max_events": self.max_events,
            "max_metrics": self.max_metrics,
            "current_events": len(self.events),
            "current_metrics": len(self.metrics),
            "current_sessions": len(self.user_sessions),
            "cleanup_interval": self.cleanup_interval,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de analytics."""
        try:
            return {
                "status": "healthy",
                "total_events": len(self.events),
                "total_metrics": len(self.metrics),
                "total_sessions": len(self.user_sessions),
                "active_sessions": len([s for s in self.user_sessions.values() if s.is_active]),
                "retention_days": self.retention_days,
                "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de analytics: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




