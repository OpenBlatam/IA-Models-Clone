"""
Servicio para dashboard de métricas avanzadas
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .analytics_service import analytics_service
from .history_service import HistoryService
from .favorites_service import FavoritesService

logger = logging.getLogger(__name__)


class DashboardService:
    """Servicio para generar dashboards de métricas"""
    
    def __init__(self):
        self.analytics = analytics_service
        self.history = HistoryService()
        self.favorites = FavoritesService()
        self.logger = logger
    
    def get_comprehensive_dashboard(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Genera un dashboard completo con todas las métricas"""
        return {
            "system_metrics": self.analytics.get_stats(),
            "user_metrics": self._get_user_metrics(user_id) if user_id else None,
            "usage_trends": self._get_usage_trends(user_id),
            "popular_content": self._get_popular_content(user_id),
            "performance_metrics": self._get_performance_metrics(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_user_metrics(self, user_id: str) -> Dict[str, Any]:
        """Obtiene métricas del usuario"""
        history_stats = self.history.get_stats(user_id)
        favorites_stats = self.favorites.get_stats(user_id)
        
        return {
            "analyses_count": history_stats.get("total_analyses", 0),
            "unique_tracks": history_stats.get("unique_tracks", 0),
            "favorites_count": favorites_stats.get("total_favorites", 0),
            "most_analyzed_genre": history_stats.get("most_analyzed_genre"),
            "first_analysis": history_stats.get("first_analysis"),
            "last_analysis": history_stats.get("last_analysis")
        }
    
    def _get_usage_trends(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene tendencias de uso"""
        history = self.history.get_history(user_id, limit=100)
        
        # Agrupar por día
        daily_usage = defaultdict(int)
        for entry in history:
            date = entry.get("timestamp", "")[:10]  # YYYY-MM-DD
            if date:
                daily_usage[date] += 1
        
        # Últimos 7 días
        last_7_days = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            last_7_days.append({
                "date": date,
                "count": daily_usage.get(date, 0)
            })
        
        last_7_days.reverse()
        
        return {
            "daily_usage": last_7_days,
            "total_days_active": len(daily_usage),
            "average_per_day": sum(daily_usage.values()) / max(len(daily_usage), 1)
        }
    
    def _get_popular_content(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene contenido más popular"""
        history = self.history.get_history(user_id, limit=100)
        
        # Tracks más analizados
        track_counts = defaultdict(int)
        for entry in history:
            track_id = entry.get("track_id")
            if track_id:
                track_counts[track_id] += 1
        
        top_tracks = sorted(track_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Géneros más analizados
        genre_counts = defaultdict(int)
        for entry in history:
            genre = entry.get("analysis_summary", {}).get("genre")
            if genre:
                genre_counts[genre] += 1
        
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "most_analyzed_tracks": [
                {"track_id": track_id, "count": count}
                for track_id, count in top_tracks
            ],
            "most_analyzed_genres": [
                {"genre": genre, "count": count}
                for genre, count in top_genres
            ]
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento"""
        stats = self.analytics.get_stats()
        
        avg_response_times = stats.get("average_response_times", {})
        
        # Calcular métricas agregadas
        all_times = []
        for endpoint_stats in avg_response_times.values():
            all_times.extend([
                endpoint_stats.get("avg", 0),
                endpoint_stats.get("min", 0),
                endpoint_stats.get("max", 0)
            ])
        
        return {
            "average_response_time": sum(all_times) / len(all_times) if all_times else 0,
            "fastest_endpoint": min(
                avg_response_times.items(),
                key=lambda x: x[1].get("avg", float('inf'))
            )[0] if avg_response_times else None,
            "slowest_endpoint": max(
                avg_response_times.items(),
                key=lambda x: x[1].get("avg", 0)
            )[0] if avg_response_times else None,
            "error_rate": stats.get("error_rate", 0),
            "requests_per_minute": stats.get("requests_per_minute", 0)
        }

