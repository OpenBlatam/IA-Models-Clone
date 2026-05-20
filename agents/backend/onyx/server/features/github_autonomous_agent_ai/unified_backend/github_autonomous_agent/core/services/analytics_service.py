"""
Servicio de Analytics y Análisis de Datos.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class AnalyticsEvent:
    """Evento de analytics."""
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AnalyticsService:
    """
    Servicio de analytics y análisis de datos con mejoras.
    
    Attributes:
        events: Lista de eventos de analytics
        max_events: Número máximo de eventos en memoria
        stats: Estadísticas de eventos
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Inicializar servicio de analytics con validaciones.
        
        Args:
            max_events: Número máximo de eventos a mantener en memoria (debe ser entero positivo)
            
        Raises:
            ValueError: Si max_events es inválido
        """
        # Validación
        if not isinstance(max_events, int) or max_events < 1:
            raise ValueError(f"max_events debe ser un entero positivo, recibido: {max_events}")
        
        self.events: List[AnalyticsEvent] = []
        self.max_events = max_events
        self.stats = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_user": defaultdict(int),
            "events_today": 0
        }
        
        logger.info(f"✅ AnalyticsService inicializado: max_events={max_events}")
    
    def track_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar evento de analytics con validaciones.
        
        Args:
            event_type: Tipo de evento (debe ser string no vacío)
            user_id: ID del usuario (opcional, debe ser string si se proporciona)
            session_id: ID de sesión (opcional, debe ser string si se proporciona)
            properties: Propiedades adicionales (opcional, debe ser diccionario si se proporciona)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not event_type or not isinstance(event_type, str) or not event_type.strip():
            raise ValueError(f"event_type debe ser un string no vacío, recibido: {event_type}")
        
        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(f"user_id debe ser un string no vacío si se proporciona, recibido: {user_id}")
            user_id = user_id.strip()
        
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError(f"session_id debe ser un string no vacío si se proporciona, recibido: {session_id}")
            session_id = session_id.strip()
        
        if properties is not None:
            if not isinstance(properties, dict):
                raise ValueError(f"properties debe ser un diccionario si se proporciona, recibido: {type(properties)}")
        
        event_type = event_type.strip()
        
        try:
            event = AnalyticsEvent(
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                properties=properties or {}
            )
            
            self.events.append(event)
            self.stats["total_events"] += 1
            self.stats["events_by_type"][event_type] += 1
            
            if user_id:
                self.stats["events_by_user"][user_id] += 1
            
            # Contar eventos de hoy
            if event.timestamp.date() == datetime.now().date():
                self.stats["events_today"] += 1
            
            # Limitar tamaño
            if len(self.events) > self.max_events:
                removed = self.events.pop(0)
                logger.debug(f"Evento antiguo removido: {removed.event_type}")
            
            logger.debug(
                f"📊 Event tracked: {event_type} (user: {user_id or 'N/A'}, "
                f"session: {session_id or 'N/A'}, properties: {len(properties or {})})"
            )
        except Exception as e:
            logger.error(f"Error al registrar evento de analytics: {e}", exc_info=True)
            raise ValueError(f"Error al registrar evento: {e}") from e
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[AnalyticsEvent]:
        """
        Obtener eventos filtrados con validaciones.
        
        Args:
            event_type: Filtrar por tipo (opcional, debe ser string si se proporciona)
            user_id: Filtrar por usuario (opcional, debe ser string si se proporciona)
            start_date: Fecha de inicio (opcional, debe ser datetime si se proporciona)
            end_date: Fecha de fin (opcional, debe ser datetime si se proporciona)
            limit: Límite de resultados (opcional, debe ser entero positivo)
            
        Returns:
            Lista de eventos filtrados y ordenados
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if event_type is not None:
            if not isinstance(event_type, str) or not event_type.strip():
                raise ValueError(f"event_type debe ser un string no vacío si se proporciona, recibido: {event_type}")
            event_type = event_type.strip()
        
        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(f"user_id debe ser un string no vacío si se proporciona, recibido: {user_id}")
            user_id = user_id.strip()
        
        if start_date is not None:
            if not isinstance(start_date, datetime):
                raise ValueError(f"start_date debe ser un datetime si se proporciona, recibido: {type(start_date)}")
        
        if end_date is not None:
            if not isinstance(end_date, datetime):
                raise ValueError(f"end_date debe ser un datetime si se proporciona, recibido: {type(end_date)}")
        
        if start_date and end_date and start_date > end_date:
            raise ValueError(f"start_date debe ser anterior a end_date")
        
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"limit debe ser un entero positivo si se proporciona, recibido: {limit}")
            if limit > 10000:
                logger.warning(f"limit muy alto ({limit}), limitando a 10000")
                limit = 10000
        
        filtered = self.events.copy()
        initial_count = len(filtered)
        
        if event_type:
            before = len(filtered)
            filtered = [e for e in filtered if e.event_type == event_type]
            logger.debug(f"Filtro por event_type '{event_type}': {before} -> {len(filtered)}")
        
        if user_id:
            before = len(filtered)
            filtered = [e for e in filtered if e.user_id == user_id]
            logger.debug(f"Filtro por user_id '{user_id}': {before} -> {len(filtered)}")
        
        if start_date:
            before = len(filtered)
            filtered = [e for e in filtered if e.timestamp >= start_date]
            logger.debug(f"Filtro por start_date '{start_date}': {before} -> {len(filtered)}")
        
        if end_date:
            before = len(filtered)
            filtered = [e for e in filtered if e.timestamp <= end_date]
            logger.debug(f"Filtro por end_date '{end_date}': {before} -> {len(filtered)}")
        
        # Ordenar por timestamp (más recientes primero)
        filtered.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            filtered = filtered[:limit]
        
        logger.debug(
            f"Eventos obtenidos: {len(filtered)}/{initial_count} "
            f"(filtros: event_type={event_type or 'None'}, user_id={user_id or 'None'}, "
            f"start_date={start_date or 'None'}, end_date={end_date or 'None'}, limit={limit or 'None'})"
        )
        
        return filtered
    
    def get_event_counts(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Obtener conteos de eventos por tipo.
        
        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Diccionario con conteos por tipo
        """
        events = self.get_events(start_date=start_date, end_date=end_date)
        counts = defaultdict(int)
        
        for event in events:
            counts[event.event_type] += 1
        
        return dict(counts)
    
    def get_user_activity(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Obtener actividad de un usuario.
        
        Args:
            user_id: ID del usuario
            days: Número de días hacia atrás
            
        Returns:
            Actividad del usuario
        """
        start_date = datetime.now() - timedelta(days=days)
        events = self.get_events(user_id=user_id, start_date=start_date)
        
        activity_by_day = defaultdict(int)
        activity_by_type = defaultdict(int)
        
        for event in events:
            day = event.timestamp.date().isoformat()
            activity_by_day[day] += 1
            activity_by_type[event.event_type] += 1
        
        return {
            "user_id": user_id,
            "total_events": len(events),
            "events_by_day": dict(activity_by_day),
            "events_by_type": dict(activity_by_type),
            "period_days": days
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            **self.stats,
            "current_events_count": len(self.events),
            "max_events": self.max_events,
            "top_event_types": dict(
                sorted(
                    self.stats["events_by_type"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "top_users": dict(
                sorted(
                    self.stats["events_by_user"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )
        }
    
    def clear_old_events(self, days: int = 30) -> int:
        """
        Limpiar eventos antiguos con validaciones.
        
        Args:
            days: Días hacia atrás para mantener (debe ser entero positivo)
            
        Returns:
            Número de eventos eliminados
            
        Raises:
            ValueError: Si days es inválido
        """
        # Validación
        if not isinstance(days, int) or days < 1:
            raise ValueError(f"days debe ser un entero positivo, recibido: {days}")
        
        cutoff = datetime.now() - timedelta(days=days)
        initial_count = len(self.events)
        
        self.events = [e for e in self.events if e.timestamp >= cutoff]
        
        removed = initial_count - len(self.events)
        
        logger.info(
            f"🧹 Eliminados {removed} eventos antiguos (más de {days} días): "
            f"{initial_count} -> {len(self.events)} eventos"
        )
        
        return removed

