"""
Analytics Engine - Sistema de analytics avanzado
=================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Motor de analytics avanzado con análisis predictivo.
    """
    
    def __init__(self, analytics_dir: str = "data/analytics"):
        """
        Inicializar motor de analytics.
        
        Args:
            analytics_dir: Directorio para analytics
        """
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        self.events: List[Dict[str, Any]] = []
        self.aggregated_data: Dict[str, Any] = defaultdict(lambda: defaultdict(int))
    
    def track_event(
        self,
        event_type: str,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """
        Registra un evento.
        
        Args:
            event_type: Tipo de evento
            properties: Propiedades del evento
            user_id: ID del usuario (opcional)
        """
        event = {
            "event_type": event_type,
            "properties": properties or {},
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.events.append(event)
        
        # Agregar a datos agregados
        self.aggregated_data[event_type]["total"] += 1
        if user_id:
            self.aggregated_data[event_type]["users"] += 1
        
        # Mantener solo últimos 10000 eventos
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
        
        logger.debug(f"Evento registrado: {event_type}")
    
    def get_trends(
        self,
        event_type: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Obtiene tendencias de un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            days: Días a analizar
            
        Returns:
            Tendencias
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_events = [
            e for e in self.events
            if e["event_type"] == event_type and
            datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        
        # Agrupar por día
        daily_counts = defaultdict(int)
        for event in recent_events:
            date = datetime.fromisoformat(event["timestamp"]).date()
            daily_counts[str(date)] += 1
        
        return {
            "event_type": event_type,
            "period_days": days,
            "total_events": len(recent_events),
            "daily_breakdown": dict(daily_counts),
            "average_per_day": round(len(recent_events) / days, 2) if days > 0 else 0
        }
    
    def get_user_analytics(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtiene analytics de un usuario.
        
        Args:
            user_id: ID del usuario
            days: Días a analizar
            
        Returns:
            Analytics del usuario
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        user_events = [
            e for e in self.events
            if e.get("user_id") == user_id and
            datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        
        # Agrupar por tipo de evento
        event_counts = defaultdict(int)
        for event in user_events:
            event_counts[event["event_type"]] += 1
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_events": len(user_events),
            "events_by_type": dict(event_counts),
            "most_active_day": self._get_most_active_day(user_events)
        }
    
    def predict_usage(
        self,
        event_type: str,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Predice uso futuro basándose en tendencias.
        
        Args:
            event_type: Tipo de evento
            days_ahead: Días a predecir
            
        Returns:
            Predicción
        """
        trends = self.get_trends(event_type, days=30)
        
        avg_per_day = trends.get("average_per_day", 0)
        predicted_total = avg_per_day * days_ahead
        
        return {
            "event_type": event_type,
            "days_ahead": days_ahead,
            "predicted_events": round(predicted_total),
            "confidence": "medium",  # En producción sería calculado
            "based_on_days": 30
        }
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """
        Genera insights automáticos.
        
        Returns:
            Lista de insights
        """
        insights = []
        
        # Insight: Eventos más comunes
        event_counts = defaultdict(int)
        for event in self.events[-1000:]:  # Últimos 1000 eventos
            event_counts[event["event_type"]] += 1
        
        if event_counts:
            most_common = max(event_counts.items(), key=lambda x: x[1])
            insights.append({
                "type": "most_common_event",
                "message": f"Evento más común: {most_common[0]} ({most_common[1]} veces)",
                "priority": "info"
            })
        
        # Insight: Tendencias
        for event_type in list(event_counts.keys())[:5]:
            trends = self.get_trends(event_type, days=7)
            if trends["total_events"] > 0:
                insights.append({
                    "type": "trend",
                    "message": f"{event_type}: {trends['total_events']} eventos en últimos 7 días",
                    "priority": "info"
                })
        
        return insights
    
    def _get_most_active_day(self, events: List[Dict[str, Any]]) -> Optional[str]:
        """Obtiene el día más activo"""
        if not events:
            return None
        
        daily_counts = defaultdict(int)
        for event in events:
            date = datetime.fromisoformat(event["timestamp"]).date()
            daily_counts[str(date)] += 1
        
        if daily_counts:
            return max(daily_counts.items(), key=lambda x: x[1])[0]
        return None
    
    def export_analytics(
        self,
        format: str = "json"
    ) -> str:
        """
        Exporta analytics.
        
        Args:
            format: Formato de exportación
            
        Returns:
            Ruta al archivo exportado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_{timestamp}.json"
        filepath = self.analytics_dir / filename
        
        analytics_data = {
            "exported_at": datetime.now().isoformat(),
            "total_events": len(self.events),
            "aggregated_data": dict(self.aggregated_data),
            "recent_events": self.events[-1000:]  # Últimos 1000
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analytics exportados: {filepath}")
        return str(filepath)




