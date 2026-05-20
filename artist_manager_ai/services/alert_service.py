"""
Alert Service
=============

Servicio de alertas inteligentes.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Tipos de alertas."""
    CONFLICT = "conflict"
    OVERDUE = "overdue"
    LOW_COMPLETION = "low_completion"
    PROTOCOL_VIOLATION = "protocol_violation"
    SCHEDULE_OVERLOAD = "schedule_overload"
    MISSING_PREPARATION = "missing_preparation"


class AlertService:
    """Servicio de alertas."""
    
    def __init__(self):
        """Inicializar servicio de alertas."""
        self.alerts: Dict[str, List[Dict[str, Any]]] = {}
        self._logger = logger
    
    def check_conflicts(
        self,
        artist_id: str,
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verificar conflictos de horario.
        
        Args:
            artist_id: ID del artista
            events: Lista de eventos
        
        Returns:
            Lista de conflictos
        """
        conflicts = []
        sorted_events = sorted(
            events,
            key=lambda e: datetime.fromisoformat(e.get("start_time", ""))
        )
        
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            current_end = datetime.fromisoformat(current.get("end_time", ""))
            next_start = datetime.fromisoformat(next_event.get("start_time", ""))
            
            if current_end > next_start:
                conflicts.append({
                    "type": AlertType.CONFLICT.value,
                    "artist_id": artist_id,
                    "event1": current.get("id"),
                    "event2": next_event.get("id"),
                    "message": f"Conflicto entre '{current.get('title')}' y '{next_event.get('title')}'",
                    "severity": "high",
                    "detected_at": datetime.now().isoformat()
                })
        
        return conflicts
    
    def check_overdue_routines(
        self,
        artist_id: str,
        routines: List[Dict[str, Any]],
        completions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verificar rutinas vencidas.
        
        Args:
            artist_id: ID del artista
            routines: Lista de rutinas
            completions: Completaciones recientes
        
        Returns:
            Lista de alertas
        """
        alerts = []
        today = datetime.now().date()
        
        for routine in routines:
            routine_id = routine.get("id")
            
            # Verificar si fue completada hoy
            today_completions = [
                c for c in completions
                if c.get("task_id") == routine_id
                and datetime.fromisoformat(c.get("completed_at", "")).date() == today
                and c.get("status") == "completed"
            ]
            
            if not today_completions:
                # Verificar si ya pasó la hora programada
                try:
                    scheduled_time = datetime.strptime(
                        routine.get("scheduled_time", ""),
                        "%H:%M:%S"
                    ).time()
                    current_time = datetime.now().time()
                    
                    if current_time > scheduled_time:
                        alerts.append({
                            "type": AlertType.OVERDUE.value,
                            "artist_id": artist_id,
                            "routine_id": routine_id,
                            "routine_title": routine.get("title"),
                            "message": f"Rutina '{routine.get('title')}' está vencida",
                            "severity": "medium",
                            "detected_at": datetime.now().isoformat()
                        })
                except:
                    pass
        
        return alerts
    
    def check_low_completion_rate(
        self,
        artist_id: str,
        routine_id: str,
        completion_rate: float,
        threshold: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Verificar tasa de completación baja.
        
        Args:
            artist_id: ID del artista
            routine_id: ID de la rutina
            completion_rate: Tasa de completación
            threshold: Umbral mínimo
        
        Returns:
            Alerta si la tasa es baja
        """
        if completion_rate < threshold:
            return {
                "type": AlertType.LOW_COMPLETION.value,
                "artist_id": artist_id,
                "routine_id": routine_id,
                "completion_rate": completion_rate,
                "threshold": threshold,
                "message": f"Tasa de completación baja: {completion_rate:.1%} (umbral: {threshold:.1%})",
                "severity": "medium",
                "detected_at": datetime.now().isoformat()
            }
        return None
    
    def check_schedule_overload(
        self,
        artist_id: str,
        events: List[Dict[str, Any]],
        max_events_per_day: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Verificar sobrecarga de agenda.
        
        Args:
            artist_id: ID del artista
            events: Lista de eventos
            max_events_per_day: Máximo de eventos por día
        
        Returns:
            Lista de alertas
        """
        alerts = []
        
        # Agrupar eventos por día
        events_by_date = {}
        for event in events:
            event_date = datetime.fromisoformat(event.get("start_time", "")).date()
            if event_date not in events_by_date:
                events_by_date[event_date] = []
            events_by_date[event_date].append(event)
        
        # Verificar días con muchos eventos
        for date, day_events in events_by_date.items():
            if len(day_events) > max_events_per_day:
                alerts.append({
                    "type": AlertType.SCHEDULE_OVERLOAD.value,
                    "artist_id": artist_id,
                    "date": date.isoformat(),
                    "event_count": len(day_events),
                    "max_recommended": max_events_per_day,
                    "message": f"Sobrecarga de agenda el {date}: {len(day_events)} eventos",
                    "severity": "high",
                    "detected_at": datetime.now().isoformat()
                })
        
        return alerts
    
    def get_all_alerts(self, artist_id: str) -> List[Dict[str, Any]]:
        """
        Obtener todas las alertas de un artista.
        
        Args:
            artist_id: ID del artista
        
        Returns:
            Lista de alertas
        """
        return self.alerts.get(artist_id, [])
    
    def add_alert(self, artist_id: str, alert: Dict[str, Any]):
        """
        Agregar alerta.
        
        Args:
            artist_id: ID del artista
            alert: Alerta a agregar
        """
        if artist_id not in self.alerts:
            self.alerts[artist_id] = []
        
        self.alerts[artist_id].append(alert)
        self._logger.info(f"Alert added for artist {artist_id}: {alert.get('type')}")




