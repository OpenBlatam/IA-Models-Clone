"""
Servicio de Dashboard - Datos agregados para visualización
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..analytics.analytics_service import AnalyticsService
from ..services.storage_service import StorageService
from ..db.models import IdentityProfileModel, GeneratedContentModel, ScheduledTaskModel
from ..db.base import get_db_session
from sqlalchemy import func

logger = logging.getLogger(__name__)


class DashboardService:
    """Servicio para datos del dashboard"""
    
    def __init__(self):
        self.analytics = AnalyticsService()
        self.storage = StorageService()
    
    def get_dashboard_data(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene datos completos para dashboard
        
        Args:
            user_id: ID de usuario (opcional, para multi-user)
            
        Returns:
            Datos del dashboard
        """
        with get_db_session() as db:
            # Estadísticas generales
            total_identities = db.query(func.count(IdentityProfileModel.id)).scalar() or 0
            total_content = db.query(func.count(GeneratedContentModel.id)).scalar() or 0
            
            # Contenido por plataforma (últimos 30 días)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_content = db.query(
                GeneratedContentModel.platform,
                func.count(GeneratedContentModel.id).label("count")
            ).filter(
                GeneratedContentModel.generated_at >= thirty_days_ago
            ).group_by(GeneratedContentModel.platform).all()
            
            # Identidades recientes
            recent_identities = db.query(func.count(IdentityProfileModel.id)).filter(
                IdentityProfileModel.created_at >= thirty_days_ago
            ).scalar() or 0
            
            # Tareas programadas activas
            active_schedules = db.query(func.count(ScheduledTaskModel.id)).filter(
                ScheduledTaskModel.enabled == True
            ).scalar() or 0
            
            # Contenido generado hoy
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            content_today = db.query(func.count(GeneratedContentModel.id)).filter(
                GeneratedContentModel.generated_at >= today_start
            ).scalar() or 0
            
            return {
                "overview": {
                    "total_identities": total_identities,
                    "total_content": total_content,
                    "recent_identities_30d": recent_identities,
                    "content_today": content_today,
                    "active_schedules": active_schedules
                },
                "content_by_platform": {
                    platform: count for platform, count in recent_content
                },
                "recent_activity": self._get_recent_activity(db),
                "top_identities": self._get_top_identities(db),
                "system_health": self._get_system_health()
            }
    
    def _get_recent_activity(self, db) -> List[Dict[str, Any]]:
        """Obtiene actividad reciente"""
        activities = []
        
        # Identidades creadas recientemente
        recent_identities = db.query(IdentityProfileModel).order_by(
            IdentityProfileModel.created_at.desc()
        ).limit(5).all()
        
        for identity in recent_identities:
            activities.append({
                "type": "identity_created",
                "timestamp": identity.created_at.isoformat(),
                "data": {
                    "identity_id": identity.id,
                    "username": identity.username
                }
            })
        
        # Contenido generado recientemente
        recent_content = db.query(GeneratedContentModel).order_by(
            GeneratedContentModel.generated_at.desc()
        ).limit(5).all()
        
        for content in recent_content:
            activities.append({
                "type": "content_generated",
                "timestamp": content.generated_at.isoformat(),
                "data": {
                    "content_id": content.id,
                    "platform": content.platform,
                    "identity_id": content.identity_profile_id
                }
            })
        
        # Ordenar por timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return activities[:10]
    
    def _get_top_identities(self, db) -> List[Dict[str, Any]]:
        """Obtiene identidades más activas"""
        # Identidades con más contenido generado
        top_identities = db.query(
            GeneratedContentModel.identity_profile_id,
            func.count(GeneratedContentModel.id).label("content_count")
        ).group_by(
            GeneratedContentModel.identity_profile_id
        ).order_by(
            func.count(GeneratedContentModel.id).desc()
        ).limit(5).all()
        
        result = []
        for identity_id, content_count in top_identities:
            identity = db.query(IdentityProfileModel).filter_by(id=identity_id).first()
            if identity:
                result.append({
                    "identity_id": identity.id,
                    "username": identity.username,
                    "content_count": content_count,
                    "total_videos": identity.total_videos,
                    "total_posts": identity.total_posts
                })
        
        return result
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Obtiene salud del sistema"""
        from ..analytics.metrics import get_metrics_collector
        
        metrics = get_metrics_collector()
        all_metrics = metrics.get_all_metrics()
        
        return {
            "uptime_seconds": all_metrics.get("uptime_seconds", 0),
            "total_requests": sum(all_metrics.get("counters", {}).values()),
            "status": "healthy"
        }

