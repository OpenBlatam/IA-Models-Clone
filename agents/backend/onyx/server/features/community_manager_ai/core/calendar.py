"""
Content Calendar - Calendario de Contenido
==========================================

Sistema de calendario para visualizar y gestionar publicaciones programadas.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContentCalendar:
    """Calendario de contenido para publicaciones"""
    
    def __init__(self):
        """Inicializar el calendario"""
        self.events: Dict[str, Dict[str, Any]] = {}
        self.daily_events: Dict[str, List[str]] = defaultdict(list)
        logger.info("Content Calendar inicializado")
    
    def add_event(
        self,
        event_id: str,
        scheduled_time: datetime,
        event_data: Dict[str, Any]
    ):
        """
        Agregar un evento al calendario
        
        Args:
            event_id: ID único del evento
            scheduled_time: Fecha y hora programada
            event_data: Datos del evento
        """
        date_key = scheduled_time.date().isoformat()
        
        self.events[event_id] = {
            "id": event_id,
            "scheduled_time": scheduled_time,
            "date_key": date_key,
            **event_data
        }
        
        self.daily_events[date_key].append(event_id)
        logger.info(f"Evento agregado al calendario: {event_id} para {date_key}")
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener eventos en un rango de fechas
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Lista de eventos ordenados por fecha
        """
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=30)
        
        events = []
        
        for event_id, event in self.events.items():
            scheduled_time = event.get("scheduled_time")
            if scheduled_time and start_date <= scheduled_time <= end_date:
                events.append(event)
        
        # Ordenar por fecha
        events.sort(key=lambda x: x.get("scheduled_time", datetime.min))
        
        return events
    
    def get_daily_events(self, date: datetime) -> List[Dict[str, Any]]:
        """
        Obtener eventos de un día específico
        
        Args:
            date: Fecha del día
            
        Returns:
            Lista de eventos del día
        """
        date_key = date.date().isoformat()
        event_ids = self.daily_events.get(date_key, [])
        
        events = []
        for event_id in event_ids:
            if event_id in self.events:
                events.append(self.events[event_id])
        
        # Ordenar por hora
        events.sort(key=lambda x: x.get("scheduled_time", datetime.min))
        
        return events
    
    def get_weekly_view(
        self,
        start_date: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtener vista semanal del calendario
        
        Args:
            start_date: Fecha de inicio de la semana
            
        Returns:
            Dict con eventos por día de la semana
        """
        if not start_date:
            start_date = datetime.now()
        
        # Ajustar al inicio de la semana (lunes)
        days_since_monday = start_date.weekday()
        week_start = start_date - timedelta(days=days_since_monday)
        
        weekly_view = {}
        
        for i in range(7):
            day = week_start + timedelta(days=i)
            day_key = day.strftime("%Y-%m-%d")
            weekly_view[day_key] = self.get_daily_events(day)
        
        return weekly_view
    
    def remove_event(self, event_id: str) -> bool:
        """
        Remover un evento del calendario
        
        Args:
            event_id: ID del evento
            
        Returns:
            True si se removió exitosamente
        """
        if event_id in self.events:
            event = self.events[event_id]
            date_key = event.get("date_key")
            
            if date_key and event_id in self.daily_events.get(date_key, []):
                self.daily_events[date_key].remove(event_id)
            
            del self.events[event_id]
            logger.info(f"Evento removido del calendario: {event_id}")
            return True
        
        return False
    
    def get_monthly_view(
        self,
        year: int,
        month: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtener vista mensual del calendario
        
        Args:
            year: Año
            month: Mes (1-12)
            
        Returns:
            Dict con eventos por día del mes
        """
        from calendar import monthrange
        
        _, last_day = monthrange(year, month)
        monthly_view = {}
        
        for day in range(1, last_day + 1):
            date = datetime(year, month, day)
            date_key = date.strftime("%Y-%m-%d")
            monthly_view[date_key] = self.get_daily_events(date)
        
        return monthly_view
    
    def get_platform_events(
        self,
        platform: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener eventos de una plataforma específica
        
        Args:
            platform: Nombre de la plataforma
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Lista de eventos de la plataforma
        """
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=30)
        
        platform_events = []
        
        for event in self.events.values():
            platforms = event.get("platforms", [])
            scheduled_time = event.get("scheduled_time")
            
            if platform in platforms and scheduled_time:
                if start_date <= scheduled_time <= end_date:
                    platform_events.append(event)
        
        platform_events.sort(key=lambda x: x.get("scheduled_time", datetime.min))
        return platform_events
    
    def get_busy_days(
        self,
        threshold: int = 5
    ) -> List[str]:
        """
        Obtener días con más eventos (días ocupados)
        
        Args:
            threshold: Número mínimo de eventos para considerar un día ocupado
            
        Returns:
            Lista de fechas (ISO format) de días ocupados
        """
        busy_days = []
        
        for date_key, event_ids in self.daily_events.items():
            if len(event_ids) >= threshold:
                busy_days.append(date_key)
        
        return sorted(busy_days)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del calendario
        
        Returns:
            Dict con estadísticas
        """
        total_events = len(self.events)
        
        platform_counts = defaultdict(int)
        status_counts = defaultdict(int)
        
        for event in self.events.values():
            for platform in event.get("platforms", []):
                platform_counts[platform] += 1
            
            status = event.get("status", "unknown")
            status_counts[status] += 1
        
        return {
            "total_events": total_events,
            "platform_distribution": dict(platform_counts),
            "status_distribution": dict(status_counts),
            "busy_days": len(self.get_busy_days()),
            "days_with_events": len(self.daily_events)
        }



