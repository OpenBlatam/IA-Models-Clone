"""
Alerts endpoints for intelligent alerts system
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class AlertsRouter(BaseRouter):
    """Router for alerts endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/alerts", tags=["Alerts"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all alerts routes"""
        
        @self.router.get("/check", response_model=dict)
        @self.handle_exceptions
        async def check_alerts(
            track_id: str = Query(...),
            previous_popularity: Optional[int] = Query(None),
            user_id: Optional[str] = Query(None)
        ):
            """Verifica alertas para un track"""
            spotify_service, alert_service = self.get_services(
                "spotify_service",
                "alert_service"
            )
            
            track_info = spotify_service.get_track(track_id)
            current_popularity = track_info.get("popularity", 0)
            
            popularity_alerts = alert_service.check_popularity_alerts(
                track_id, current_popularity, previous_popularity
            )
            
            audio_features = spotify_service.get_track_audio_features(track_id)
            
            try:
                from ...services.genre_detector import GenreDetector
                from ...services.emotion_analyzer import EmotionAnalyzer
                genre_detector = GenreDetector()
                emotion_analyzer = EmotionAnalyzer()
                
                genre = genre_detector.detect_genre(audio_features)
                emotion = emotion_analyzer.analyze_emotions(audio_features)
                
                trend_alerts = alert_service.check_trend_opportunities(
                    track_id, audio_features,
                    genre.get("primary_genre", "Unknown"),
                    emotion.get("primary_emotion", "Unknown")
                )
            except ImportError:
                trend_alerts = []
            
            artists = track_info.get("artists", [])
            collaboration_alerts = alert_service.check_collaboration_alerts(track_id, artists)
            
            all_alerts = popularity_alerts + trend_alerts + collaboration_alerts
            
            return self.count_response(
                all_alerts,
                count_key="alerts_count",
                track_id=track_id,
                alerts=all_alerts
            )
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def get_alerts(
            user_id: Optional[str] = Query(None),
            alert_type: Optional[str] = Query(None),
            priority: Optional[str] = Query(None)
        ):
            """Obtiene todas las alertas con filtros"""
            alert_service = self.get_service("alert_service")
            alerts = alert_service.get_all_alerts(user_id, alert_type, priority)
            return self.count_response(
                alerts,
                count_key="alerts_count",
                alerts=alerts
            )
        
        @self.router.put("/{alert_id}/read", response_model=dict)
        @self.handle_exceptions
        async def mark_alert_read(alert_id: str):
            """Marca una alerta como leída"""
            alert_service = self.get_service("alert_service")
            success = alert_service.mark_alert_read(alert_id)
            self.require_success(success, "Alerta no encontrada", status_code=404)
            return self.success_response(None, message="Alerta marcada como leída")
        
        @self.router.delete("/{alert_id}", response_model=dict)
        @self.handle_exceptions
        async def delete_alert(alert_id: str):
            """Elimina una alerta"""
            alert_service = self.get_service("alert_service")
            success = alert_service.delete_alert(alert_id)
            self.require_success(success, "Alerta no encontrada", status_code=404)
            return self.success_response(None, message="Alerta eliminada")


def get_alerts_router() -> AlertsRouter:
    """Factory function to get alerts router"""
    return AlertsRouter()

