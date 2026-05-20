"""
Sistema de Analytics Avanzados
================================

Sistema para análisis avanzado de datos y métricas.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    """Evento de analytics"""
    event_type: str
    timestamp: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AdvancedAnalytics:
    """
    Sistema de analytics avanzados
    
    Proporciona:
    - Tracking de eventos
    - Análisis de comportamiento
    - Métricas de negocio
    - Funnels de conversión
    - Cohort analysis
    - Segmentación de usuarios
    """
    
    def __init__(self):
        """Inicializar analytics"""
        self.events: List[AnalyticsEvent] = []
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        logger.info("AdvancedAnalytics inicializado")
    
    def track_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Registrar evento"""
        event = AnalyticsEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
            user_id=user_id,
            session_id=session_id
        )
        
        self.events.append(event)
        
        # Mantener solo últimos 10000 eventos
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
        
        # Actualizar perfil de usuario
        if user_id:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "user_id": user_id,
                    "first_seen": event.timestamp,
                    "last_seen": event.timestamp,
                    "event_count": 0,
                    "events": []
                }
            
            profile = self.user_profiles[user_id]
            profile["last_seen"] = event.timestamp
            profile["event_count"] += 1
            profile["events"].append(event_type)
    
    def get_event_stats(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtener estadísticas de eventos"""
        filtered = self._filter_events(event_type, start_date, end_date)
        
        if not filtered:
            return {"count": 0}
        
        # Agrupar por día
        daily_counts = defaultdict(int)
        for event in filtered:
            date = datetime.fromisoformat(event.timestamp).date()
            daily_counts[date.isoformat()] += 1
        
        return {
            "total_count": len(filtered),
            "unique_users": len(set(e.user_id for e in filtered if e.user_id)),
            "daily_counts": dict(daily_counts),
            "avg_per_day": len(filtered) / len(daily_counts) if daily_counts else 0
        }
    
    def get_funnel_analysis(
        self,
        steps: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Análisis de funnel
        
        Args:
            steps: Lista de tipos de eventos en orden
            start_date: Fecha de inicio
            end_date: Fecha de fin
        
        Returns:
            Análisis de funnel
        """
        filtered = self._filter_events(None, start_date, end_date)
        
        # Agrupar por usuario
        user_events = defaultdict(list)
        for event in filtered:
            if event.user_id:
                user_events[event.user_id].append(event)
        
        # Contar conversiones por paso
        step_counts = {}
        for i, step in enumerate(steps):
            if i == 0:
                # Primer paso: contar todos los usuarios que lo completaron
                step_counts[step] = sum(
                    1 for events in user_events.values()
                    if any(e.event_type == step for e in events)
                )
            else:
                # Pasos siguientes: contar usuarios que completaron pasos anteriores
                prev_step = steps[i-1]
                step_counts[step] = sum(
                    1 for events in user_events.values()
                    if any(e.event_type == prev_step for e in events) and
                    any(e.event_type == step for e in events)
                )
        
        # Calcular tasas de conversión
        conversion_rates = {}
        base_count = step_counts.get(steps[0], 0)
        for step in steps:
            count = step_counts.get(step, 0)
            conversion_rates[step] = count / base_count if base_count > 0 else 0.0
        
        return {
            "steps": steps,
            "step_counts": step_counts,
            "conversion_rates": conversion_rates,
            "total_users": len(user_events)
        }
    
    def get_user_segments(
        self,
        criteria: Dict[str, Any]
    ) -> List[str]:
        """
        Segmentar usuarios según criterios
        
        Args:
            criteria: Criterios de segmentación
        
        Returns:
            Lista de user_ids que cumplen los criterios
        """
        segments = []
        
        for user_id, profile in self.user_profiles.items():
            matches = True
            
            if "min_events" in criteria:
                if profile["event_count"] < criteria["min_events"]:
                    matches = False
            
            if "event_types" in criteria:
                required_events = set(criteria["event_types"])
                user_events = set(profile["events"])
                if not required_events.issubset(user_events):
                    matches = False
            
            if matches:
                segments.append(user_id)
        
        return segments
    
    def get_cohort_analysis(
        self,
        cohort_period: str = "week"
    ) -> Dict[str, Any]:
        """
        Análisis de cohortes
        
        Args:
            cohort_period: Período de cohorte (day, week, month)
        
        Returns:
            Análisis de cohortes
        """
        cohorts = defaultdict(lambda: {"users": set(), "retention": defaultdict(int)})
        
        for user_id, profile in self.user_profiles.items():
            first_seen = datetime.fromisoformat(profile["first_seen"])
            
            # Determinar cohorte
            if cohort_period == "week":
                cohort_key = f"{first_seen.year}-W{first_seen.isocalendar()[1]}"
            elif cohort_period == "month":
                cohort_key = f"{first_seen.year}-{first_seen.month:02d}"
            else:
                cohort_key = first_seen.date().isoformat()
            
            cohorts[cohort_key]["users"].add(user_id)
        
        # Calcular retención
        for cohort_key, cohort_data in cohorts.items():
            users = cohort_data["users"]
            for user_id in users:
                profile = self.user_profiles[user_id]
                last_seen = datetime.fromisoformat(profile["last_seen"])
                first_seen = datetime.fromisoformat(profile["first_seen"])
                
                # Calcular período desde primera vista
                if cohort_period == "week":
                    periods = (last_seen - first_seen).days // 7
                elif cohort_period == "month":
                    periods = (last_seen.year - first_seen.year) * 12 + (last_seen.month - first_seen.month)
                else:
                    periods = (last_seen - first_seen).days
                
                cohort_data["retention"][periods] += 1
        
        return {
            "cohort_period": cohort_period,
            "cohorts": {
                key: {
                    "size": len(data["users"]),
                    "retention": dict(data["retention"])
                }
                for key, data in cohorts.items()
            }
        }
    
    def _filter_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[AnalyticsEvent]:
        """Filtrar eventos"""
        filtered = self.events
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if start_date:
            start = datetime.fromisoformat(start_date)
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e.timestamp) >= start
            ]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e.timestamp) <= end
            ]
        
        return filtered


# Instancia global
_advanced_analytics: Optional[AdvancedAnalytics] = None


def get_advanced_analytics() -> AdvancedAnalytics:
    """Obtener instancia global de analytics"""
    global _advanced_analytics
    if _advanced_analytics is None:
        _advanced_analytics = AdvancedAnalytics()
    return _advanced_analytics
















