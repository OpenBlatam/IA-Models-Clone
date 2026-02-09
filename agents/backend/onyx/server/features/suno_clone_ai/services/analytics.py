"""
Sistema de Analytics y Tracking

Proporciona:
- Tracking de eventos
- Análisis de uso
- Métricas de negocio
- Funnels de conversión
- Cohort analysis
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    SONG_GENERATED = "song_generated"
    SONG_PLAYED = "song_played"
    SONG_SHARED = "song_shared"
    SONG_DOWNLOADED = "song_downloaded"
    USER_REGISTERED = "user_registered"
    USER_LOGIN = "user_login"
    SEARCH_PERFORMED = "search_performed"
    PLAYLIST_CREATED = "playlist_created"
    FAVORITE_ADDED = "favorite_added"


@dataclass
class AnalyticsEvent:
    """Representa un evento de analytics"""
    event_type: EventType
    user_id: Optional[str]
    session_id: Optional[str]
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el evento a diccionario"""
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "properties": self.properties,
            "timestamp": self.timestamp.isoformat()
        }


class AnalyticsService:
    """Servicio de analytics y tracking"""
    
    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._user_events: Dict[str, List[AnalyticsEvent]] = defaultdict(list)
        self._session_events: Dict[str, List[AnalyticsEvent]] = defaultdict(list)
        logger.info("AnalyticsService initialized")
    
    def track_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Registra un evento
        
        Args:
            event_type: Tipo de evento
            user_id: ID del usuario (opcional)
            session_id: ID de sesión (opcional)
            properties: Propiedades adicionales del evento
        """
        event = AnalyticsEvent(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            properties=properties or {}
        )
        
        self.events.append(event)
        self._event_counts[event_type.value] += 1
        
        if user_id:
            self._user_events[user_id].append(event)
        
        if session_id:
            self._session_events[session_id].append(event)
        
        logger.debug(f"Event tracked: {event_type.value} for user {user_id}")
    
    def get_event_counts(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Obtiene conteos de eventos en un rango de fechas
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
        
        Returns:
            Diccionario con conteos por tipo de evento
        """
        if start_date or end_date:
            filtered_events = self._filter_events_by_date(start_date, end_date)
            counts = defaultdict(int)
            for event in filtered_events:
                counts[event.event_type.value] += 1
            return dict(counts)
        
        return dict(self._event_counts)
    
    def get_user_activity(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtiene actividad de un usuario
        
        Args:
            user_id: ID del usuario
            days: Número de días de historial
        
        Returns:
            Estadísticas de actividad del usuario
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        user_events = [
            e for e in self._user_events.get(user_id, [])
            if e.timestamp >= cutoff_date
        ]
        
        event_counts = defaultdict(int)
        for event in user_events:
            event_counts[event.event_type.value] += 1
        
        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "events_by_type": dict(event_counts),
            "first_event": user_events[0].timestamp.isoformat() if user_events else None,
            "last_event": user_events[-1].timestamp.isoformat() if user_events else None,
            "days_active": len(set(e.timestamp.date() for e in user_events))
        }
    
    def get_funnel(
        self,
        steps: List[EventType],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calcula un funnel de conversión
        
        Args:
            steps: Lista de pasos del funnel
            start_date: Fecha de inicio
            end_date: Fecha de fin
        
        Returns:
            Análisis del funnel
        """
        filtered_events = self._filter_events_by_date(start_date, end_date)
        
        # Agrupar eventos por usuario
        user_steps: Dict[str, List[EventType]] = defaultdict(list)
        for event in filtered_events:
            if event.user_id and event.event_type in steps:
                user_steps[event.user_id].append(event.event_type)
        
        # Calcular conversión por paso
        funnel_data = []
        users_at_step = set(user_steps.keys())
        
        for i, step in enumerate(steps):
            users_completed = sum(
                1 for user_events in user_steps.values()
                if step in user_events
            )
            
            conversion_rate = (
                (users_completed / len(users_at_step) * 100)
                if users_at_step else 0
            )
            
            funnel_data.append({
                "step": step.value,
                "step_number": i + 1,
                "users": users_completed,
                "conversion_rate": round(conversion_rate, 2),
                "drop_off": len(users_at_step) - users_completed
            })
            
            users_at_step = {
                user_id for user_id, events in user_steps.items()
                if step in events
            }
        
        return {
            "steps": funnel_data,
            "total_users": len(user_steps),
            "overall_conversion": (
                (len(users_at_step) / len(user_steps) * 100)
                if user_steps else 0
            )
        }
    
    def get_cohort_analysis(
        self,
        cohort_period: str = "week",
        metric: str = "song_generated"
    ) -> Dict[str, Any]:
        """
        Análisis de cohortes
        
        Args:
            cohort_period: Período de cohorte ("day", "week", "month")
            metric: Métrica a analizar
        
        Returns:
            Análisis de cohortes
        """
        # Implementación básica
        cohorts = defaultdict(lambda: defaultdict(int))
        
        for event in self.events:
            if event.event_type.value == metric and event.user_id:
                # Determinar cohorte
                if cohort_period == "week":
                    cohort_key = event.timestamp.strftime("%Y-W%W")
                elif cohort_period == "month":
                    cohort_key = event.timestamp.strftime("%Y-%m")
                else:
                    cohort_key = event.timestamp.strftime("%Y-%m-%d")
                
                cohorts[cohort_key][event.user_id] += 1
        
        return {
            "cohort_period": cohort_period,
            "metric": metric,
            "cohorts": {
                cohort: {
                    "users": len(users),
                    "total_events": sum(users.values()),
                    "avg_per_user": sum(users.values()) / len(users) if users else 0
                }
                for cohort, users in cohorts.items()
            }
        }
    
    def get_stats(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas generales
        
        Args:
            days: Número de días de historial
        
        Returns:
            Estadísticas del sistema
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_events = [
            e for e in self.events
            if e.timestamp >= cutoff_date
        ]
        
        unique_users = len(set(e.user_id for e in recent_events if e.user_id))
        unique_sessions = len(set(e.session_id for e in recent_events if e.session_id))
        
        return {
            "period_days": days,
            "total_events": len(recent_events),
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
            "events_by_type": self.get_event_counts(cutoff_date, datetime.now()),
            "avg_events_per_user": len(recent_events) / unique_users if unique_users > 0 else 0,
            "avg_events_per_session": len(recent_events) / unique_sessions if unique_sessions > 0 else 0
        }
    
    def _filter_events_by_date(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[AnalyticsEvent]:
        """Filtra eventos por rango de fechas"""
        filtered = self.events
        
        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]
        
        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]
        
        return filtered


# Instancia global
_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Obtiene la instancia global del servicio de analytics"""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service

