"""
Servicio de dashboard - Métricas en tiempo real y visualizaciones
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import Counter
import statistics


class DashboardService:
    """Servicio de dashboard y métricas en tiempo real"""
    
    def __init__(self):
        """Inicializa el servicio de dashboard"""
        pass
    
    def get_dashboard_data(
        self,
        user_id: str,
        entries: List[Dict],
        progress_data: Dict,
        analytics_data: Optional[Dict] = None
    ) -> Dict:
        """
        Genera datos completos para el dashboard
        
        Args:
            user_id: ID del usuario
            entries: Lista de entradas diarias
            progress_data: Datos de progreso
            analytics_data: Datos de análisis (opcional)
        
        Returns:
            Datos completos del dashboard
        """
        dashboard = {
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "overview": self._get_overview(progress_data),
            "charts": self._generate_charts(entries, progress_data),
            "recent_activity": self._get_recent_activity(entries),
            "quick_stats": self._get_quick_stats(progress_data, entries),
            "alerts": self._get_alerts(progress_data, analytics_data),
            "upcoming_milestones": self._get_upcoming_milestones(progress_data)
        }
        
        return dashboard
    
    def _get_overview(self, progress_data: Dict) -> Dict:
        """Obtiene resumen general"""
        return {
            "days_sober": progress_data.get("days_sober", 0),
            "time_sober_formatted": progress_data.get("time_sober_formatted", "0 días"),
            "current_streak": progress_data.get("streak_days", 0),
            "success_rate": round(progress_data.get("success_rate", 0), 2),
            "last_consumption": progress_data.get("last_consumption_date")
        }
    
    def _generate_charts(self, entries: List[Dict], progress_data: Dict) -> Dict:
        """Genera datos para gráficos"""
        if not entries:
            return {"message": "No hay datos suficientes para gráficos"}
        
        # Datos para gráfico de progreso semanal
        weekly_data = self._get_weekly_progress(entries)
        
        # Datos para gráfico de cravings
        cravings_data = self._get_cravings_data(entries)
        
        # Datos para gráfico de estado de ánimo
        mood_data = self._get_mood_data(entries)
        
        return {
            "weekly_progress": weekly_data,
            "cravings_trend": cravings_data,
            "mood_distribution": mood_data,
            "success_rate_over_time": self._get_success_rate_over_time(entries)
        }
    
    def _get_weekly_progress(self, entries: List[Dict]) -> List[Dict]:
        """Obtiene datos de progreso semanal"""
        if len(entries) < 7:
            return []
        
        weekly_data = []
        for i in range(0, len(entries), 7):
            week_entries = entries[i:i+7]
            week_data = {
                "week": len(weekly_data) + 1,
                "sober_days": sum(1 for e in week_entries if not e.get("consumed", False)),
                "total_days": len(week_entries),
                "success_rate": (sum(1 for e in week_entries if not e.get("consumed", False)) / len(week_entries) * 100) if week_entries else 0
            }
            weekly_data.append(week_data)
        
        return weekly_data[-4:]  # Últimas 4 semanas
    
    def _get_cravings_data(self, entries: List[Dict]) -> List[Dict]:
        """Obtiene datos de cravings para gráfico"""
        if not entries:
            return []
        
        # Últimos 30 días o todos si hay menos
        recent_entries = entries[-30:] if len(entries) >= 30 else entries
        
        cravings_data = []
        for entry in recent_entries:
            cravings_data.append({
                "date": entry.get("date"),
                "level": entry.get("cravings_level", 0),
                "consumed": entry.get("consumed", False)
            })
        
        return cravings_data
    
    def _get_mood_data(self, entries: List[Dict]) -> Dict:
        """Obtiene distribución de estados de ánimo"""
        if not entries:
            return {}
        
        mood_counter = Counter([e.get("mood", "neutral") for e in entries])
        return dict(mood_counter)
    
    def _get_success_rate_over_time(self, entries: List[Dict]) -> List[Dict]:
        """Obtiene tasa de éxito a lo largo del tiempo"""
        if len(entries) < 7:
            return []
        
        success_data = []
        window_size = 7  # Ventana de 7 días
        
        for i in range(window_size, len(entries) + 1, window_size):
            window_entries = entries[i-window_size:i]
            sober_count = sum(1 for e in window_entries if not e.get("consumed", False))
            success_rate = (sober_count / len(window_entries) * 100) if window_entries else 0
            
            success_data.append({
                "period": f"Semana {len(success_data) + 1}",
                "success_rate": round(success_rate, 2),
                "date_range": {
                    "start": window_entries[0].get("date") if window_entries else None,
                    "end": window_entries[-1].get("date") if window_entries else None
                }
            })
        
        return success_data[-8:]  # Últimas 8 semanas
    
    def _get_recent_activity(self, entries: List[Dict], limit: int = 5) -> List[Dict]:
        """Obtiene actividad reciente"""
        if not entries:
            return []
        
        recent = sorted(entries, key=lambda x: x.get("date", ""), reverse=True)[:limit]
        
        activity = []
        for entry in recent:
            activity.append({
                "date": entry.get("date"),
                "type": "sober_day" if not entry.get("consumed", False) else "consumption",
                "mood": entry.get("mood"),
                "cravings_level": entry.get("cravings_level", 0),
                "notes": entry.get("notes")
            })
        
        return activity
    
    def _get_quick_stats(self, progress_data: Dict, entries: List[Dict]) -> Dict:
        """Obtiene estadísticas rápidas"""
        if not entries:
            return {}
        
        avg_cravings = statistics.mean([e.get("cravings_level", 0) for e in entries]) if entries else 0
        
        return {
            "total_entries": len(entries),
            "average_cravings": round(avg_cravings, 2),
            "longest_streak": progress_data.get("longest_streak", 0),
            "milestones_achieved": len(progress_data.get("milestones_achieved", []))
        }
    
    def _get_alerts(self, progress_data: Dict, analytics_data: Optional[Dict]) -> List[Dict]:
        """Obtiene alertas importantes"""
        alerts = []
        
        # Alerta si la racha es baja
        if progress_data.get("streak_days", 0) < 3:
            alerts.append({
                "type": "warning",
                "message": "Tu racha actual es corta. Mantén el compromiso.",
                "priority": "medium"
            })
        
        # Alerta si hay tendencia negativa
        if analytics_data:
            trends = analytics_data.get("trends", {})
            if trends.get("consumption_trend") == "empeorando":
                alerts.append({
                    "type": "danger",
                    "message": "Se detectó una tendencia negativa. Considera aumentar tu apoyo.",
                    "priority": "high"
                })
        
        return alerts
    
    def _get_upcoming_milestones(self, progress_data: Dict) -> List[Dict]:
        """Obtiene próximos hitos"""
        days_sober = progress_data.get("days_sober", 0)
        milestones = [7, 30, 90, 180, 365]
        
        upcoming = []
        for milestone in milestones:
            if days_sober < milestone:
                upcoming.append({
                    "days": milestone,
                    "days_remaining": milestone - days_sober,
                    "title": f"{milestone} días de sobriedad"
                })
                if len(upcoming) >= 3:  # Solo los próximos 3
                    break
        
        return upcoming

