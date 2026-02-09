"""
Analytics Service - Sistema de analytics avanzado
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Servicio para analytics avanzado"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.user_sessions: Dict[str, List[Dict[str, Any]]] = {}
    
    def track_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        store_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Rastrear evento"""
        
        event = {
            "event_id": f"evt_{len(self.events) + 1}",
            "event_type": event_type,
            "user_id": user_id,
            "store_id": store_id,
            "properties": properties or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.events.append(event)
        
        return event
    
    def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtener analytics"""
        
        filtered_events = self._filter_events(start_date, end_date, event_type)
        
        return {
            "total_events": len(filtered_events),
            "events_by_type": self._group_by_type(filtered_events),
            "events_by_user": self._group_by_user(filtered_events),
            "events_by_store": self._group_by_store(filtered_events),
            "timeline": self._generate_timeline(filtered_events),
            "funnel": self._generate_funnel(filtered_events),
            "retention": self._calculate_retention(filtered_events)
        }
    
    def _filter_events(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        event_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Filtrar eventos"""
        filtered = self.events
        
        if start_date:
            filtered = [e for e in filtered if datetime.fromisoformat(e["timestamp"]) >= start_date]
        
        if end_date:
            filtered = [e for e in filtered if datetime.fromisoformat(e["timestamp"]) <= end_date]
        
        if event_type:
            filtered = [e for e in filtered if e["event_type"] == event_type]
        
        return filtered
    
    def _group_by_type(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Agrupar eventos por tipo"""
        grouped = defaultdict(int)
        for event in events:
            grouped[event["event_type"]] += 1
        return dict(grouped)
    
    def _group_by_user(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Agrupar eventos por usuario"""
        grouped = defaultdict(int)
        for event in events:
            if event.get("user_id"):
                grouped[event["user_id"]] += 1
        return dict(grouped)
    
    def _group_by_store(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Agrupar eventos por diseño"""
        grouped = defaultdict(int)
        for event in events:
            if event.get("store_id"):
                grouped[event["store_id"]] += 1
        return dict(grouped)
    
    def _generate_timeline(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Generar timeline de eventos"""
        timeline = defaultdict(int)
        
        for event in events:
            date = datetime.fromisoformat(event["timestamp"]).date()
            date_key = date.isoformat()
            timeline[date_key] += 1
        
        return dict(timeline)
    
    def _generate_funnel(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generar funnel de conversión"""
        funnel_steps = {
            "design_created": 0,
            "analysis_completed": 0,
            "feedback_received": 0,
            "design_approved": 0,
            "exported": 0
        }
        
        for event in events:
            event_type = event["event_type"]
            if event_type in funnel_steps:
                funnel_steps[event_type] += 1
        
        # Calcular tasas de conversión
        conversion_rates = {}
        previous_count = None
        for step, count in funnel_steps.items():
            if previous_count is not None and previous_count > 0:
                conversion_rates[step] = round((count / previous_count) * 100, 2)
            else:
                conversion_rates[step] = 100.0
            previous_count = count
        
        return {
            "steps": funnel_steps,
            "conversion_rates": conversion_rates
        }
    
    def _calculate_retention(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcular retención de usuarios"""
        # Simplificado - en producción usar cálculo más sofisticado
        user_events = defaultdict(list)
        
        for event in events:
            if event.get("user_id"):
                user_events[event["user_id"]].append(event)
        
        # Usuarios que volvieron (más de 1 evento)
        returning_users = sum(1 for events_list in user_events.values() if len(events_list) > 1)
        total_users = len(user_events)
        
        retention_rate = (returning_users / total_users * 100) if total_users > 0 else 0
        
        return {
            "total_users": total_users,
            "returning_users": returning_users,
            "retention_rate": round(retention_rate, 2)
        }
    
    def get_user_journey(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Obtener journey de usuario"""
        user_events = [e for e in self.events if e.get("user_id") == user_id]
        
        if not user_events:
            return {"message": "No hay eventos para este usuario"}
        
        # Ordenar por timestamp
        user_events.sort(key=lambda x: x["timestamp"])
        
        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "first_event": user_events[0] if user_events else None,
            "last_event": user_events[-1] if user_events else None,
            "journey": user_events,
            "time_span_days": self._calculate_time_span(user_events)
        }
    
    def _calculate_time_span(self, events: List[Dict[str, Any]]) -> float:
        """Calcular span de tiempo"""
        if len(events) < 2:
            return 0.0
        
        first = datetime.fromisoformat(events[0]["timestamp"])
        last = datetime.fromisoformat(events[-1]["timestamp"])
        
        return (last - first).total_seconds() / 86400  # días




