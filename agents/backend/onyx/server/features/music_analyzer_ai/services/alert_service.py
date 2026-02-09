"""
Servicio de alertas inteligentes para música
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Tipos de alertas"""
    POPULARITY_DROP = "popularity_drop"
    POPULARITY_SURGE = "popularity_surge"
    NEW_COLLABORATION = "new_collaboration"
    SIMILAR_TRACK_FOUND = "similar_track_found"
    TREND_OPPORTUNITY = "trend_opportunity"
    QUALITY_ISSUE = "quality_issue"


class AlertPriority(Enum):
    """Prioridades de alertas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertService:
    """Servicio de alertas inteligentes"""
    
    def __init__(self):
        self.logger = logger
        self.alerts = []  # En producción, usar base de datos
    
    def check_popularity_alerts(self, track_id: str, current_popularity: int,
                                previous_popularity: Optional[int] = None) -> List[Dict[str, Any]]:
        """Verifica alertas de popularidad"""
        alerts = []
        
        if previous_popularity is not None:
            change = current_popularity - previous_popularity
            change_percent = (change / previous_popularity * 100) if previous_popularity > 0 else 0
            
            # Alerta de caída
            if change < -10 or change_percent < -20:
                alerts.append({
                    "type": AlertType.POPULARITY_DROP.value,
                    "priority": AlertPriority.HIGH.value if change < -20 else AlertPriority.MEDIUM.value,
                    "track_id": track_id,
                    "message": f"Popularidad cayó {abs(change)} puntos ({abs(change_percent):.1f}%)",
                    "current": current_popularity,
                    "previous": previous_popularity,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Alerta de aumento
            if change > 10 or change_percent > 20:
                alerts.append({
                    "type": AlertType.POPULARITY_SURGE.value,
                    "priority": AlertPriority.MEDIUM.value,
                    "track_id": track_id,
                    "message": f"Popularidad aumentó {change} puntos ({change_percent:.1f}%)",
                    "current": current_popularity,
                    "previous": previous_popularity,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Alerta de popularidad muy baja
        if current_popularity < 20:
            alerts.append({
                "type": AlertType.QUALITY_ISSUE.value,
                "priority": AlertPriority.MEDIUM.value,
                "track_id": track_id,
                "message": "Popularidad muy baja - considerar estrategia de promoción",
                "current": current_popularity,
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def check_trend_opportunities(self, track_id: str, audio_features: Dict[str, Any],
                                 genre: str, emotion: str) -> List[Dict[str, Any]]:
        """Verifica oportunidades de tendencias"""
        alerts = []
        
        # Oportunidad: características comerciales
        danceability = audio_features.get("danceability", 0.5)
        energy = audio_features.get("energy", 0.5)
        valence = audio_features.get("valence", 0.5)
        tempo = audio_features.get("tempo", 120)
        
        # Track muy bailable y energético
        if danceability > 0.7 and energy > 0.7:
            alerts.append({
                "type": AlertType.TREND_OPPORTUNITY.value,
                "priority": AlertPriority.MEDIUM.value,
                "track_id": track_id,
                "message": "Track con alto potencial comercial - alta bailabilidad y energía",
                "factors": {
                    "danceability": danceability,
                    "energy": energy
                },
                "timestamp": datetime.now().isoformat()
            })
        
        # Track con mood positivo
        if valence > 0.7 and emotion in ["happy", "energetic"]:
            alerts.append({
                "type": AlertType.TREND_OPPORTUNITY.value,
                "priority": AlertPriority.LOW.value,
                "track_id": track_id,
                "message": "Track con mood positivo - ideal para playlists de verano/fiesta",
                "factors": {
                    "valence": valence,
                    "emotion": emotion
                },
                "timestamp": datetime.now().isoformat()
            })
        
        # Tempo comercial
        if 100 <= tempo <= 140:
            alerts.append({
                "type": AlertType.TREND_OPPORTUNITY.value,
                "priority": AlertPriority.LOW.value,
                "track_id": track_id,
                "message": "Tempo en rango comercial ideal (100-140 BPM)",
                "factors": {
                    "tempo": tempo
                },
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def check_similar_tracks(self, track_id: str, similar_tracks: List[Dict[str, Any]],
                           similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Verifica tracks muy similares"""
        alerts = []
        
        for similar in similar_tracks:
            similarity = similar.get("similarity", 0)
            if similarity >= similarity_threshold:
                alerts.append({
                    "type": AlertType.SIMILAR_TRACK_FOUND.value,
                    "priority": AlertPriority.LOW.value,
                    "track_id": track_id,
                    "message": f"Track muy similar encontrado (similitud: {similarity:.2f})",
                    "similar_track": similar,
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def check_collaboration_alerts(self, track_id: str, artists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verifica alertas de colaboraciones"""
        alerts = []
        
        if len(artists) >= 2:
            # Nueva colaboración detectada
            artist_names = [a.get("name", "Unknown") for a in artists]
            alerts.append({
                "type": AlertType.NEW_COLLABORATION.value,
                "priority": AlertPriority.LOW.value,
                "track_id": track_id,
                "message": f"Colaboración detectada: {', '.join(artist_names)}",
                "artists": artist_names,
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def get_all_alerts(self, user_id: Optional[str] = None,
                      alert_type: Optional[str] = None,
                      priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene todas las alertas con filtros"""
        alerts = self.alerts
        
        if user_id:
            alerts = [a for a in alerts if a.get("user_id") == user_id]
        
        if alert_type:
            alerts = [a for a in alerts if a.get("type") == alert_type]
        
        if priority:
            alerts = [a for a in alerts if a.get("priority") == priority]
        
        return sorted(alerts, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def mark_alert_read(self, alert_id: str) -> bool:
        """Marca una alerta como leída"""
        for alert in self.alerts:
            if alert.get("id") == alert_id:
                alert["read"] = True
                alert["read_at"] = datetime.now().isoformat()
                return True
        return False
    
    def delete_alert(self, alert_id: str) -> bool:
        """Elimina una alerta"""
        self.alerts = [a for a in self.alerts if a.get("id") != alert_id]
        return True

