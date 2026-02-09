"""
Dashboard Service - Servicio de Dashboard
==========================================

Servicio para generar datos del dashboard.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class DashboardService:
    """Servicio para datos del dashboard"""
    
    def __init__(self):
        """Inicializar servicio de dashboard"""
        logger.info("Dashboard Service inicializado")
    
    def get_overview_stats(
        self,
        manager: Any,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas generales
        
        Args:
            manager: Instancia de CommunityManager
            days: Días hacia atrás
            
        Returns:
            Dict con estadísticas
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Posts
        all_posts = manager.scheduler.get_all_posts()
        recent_posts = [
            p for p in all_posts
            if p.get("created_at") and
            datetime.fromisoformat(p.get("created_at")) >= cutoff_date
        ]
        
        # Estadísticas de posts
        status_counts = defaultdict(int)
        platform_counts = defaultdict(int)
        
        for post in recent_posts:
            status = post.get("status", "unknown")
            status_counts[status] += 1
            
            for platform in post.get("platforms", []):
                platform_counts[platform] += 1
        
        # Memes
        memes = manager.meme_manager.memes
        meme_count = len(memes)
        
        # Templates
        templates = manager.template_manager.templates if hasattr(manager, 'template_manager') else {}
        template_count = len(templates)
        
        # Plataformas conectadas
        connected_platforms = manager.social_connector.get_connected_platforms()
        
        return {
            "period_days": days,
            "posts": {
                "total": len(recent_posts),
                "by_status": dict(status_counts),
                "by_platform": dict(platform_counts)
            },
            "memes": {
                "total": meme_count
            },
            "templates": {
                "total": template_count
            },
            "platforms": {
                "connected": len(connected_platforms),
                "list": connected_platforms
            }
        }
    
    def get_engagement_summary(
        self,
        analytics_service: Any,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Obtener resumen de engagement
        
        Args:
            analytics_service: Instancia de AnalyticsService
            days: Días hacia atrás
            
        Returns:
            Dict con resumen
        """
        platforms = ["facebook", "instagram", "twitter", "linkedin", "tiktok", "youtube"]
        
        summary = {
            "total_engagement": 0,
            "total_posts": 0,
            "average_engagement_rate": 0.0,
            "by_platform": {}
        }
        
        total_engagement = 0
        total_posts = 0
        total_rates = []
        
        for platform in platforms:
            analytics = analytics_service.get_platform_analytics(platform, days)
            
            if analytics.get("total_posts", 0) > 0:
                summary["by_platform"][platform] = analytics
                total_engagement += analytics.get("total_engagement", 0)
                total_posts += analytics.get("total_posts", 0)
                total_rates.append(analytics.get("average_engagement_rate", 0))
        
        summary["total_engagement"] = total_engagement
        summary["total_posts"] = total_posts
        summary["average_engagement_rate"] = (
            sum(total_rates) / len(total_rates) if total_rates else 0.0
        )
        
        return summary
    
    def get_upcoming_posts(
        self,
        manager: Any,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtener próximos posts programados
        
        Args:
            manager: Instancia de CommunityManager
            limit: Límite de posts
            
        Returns:
            Lista de posts
        """
        now = datetime.now()
        
        all_posts = manager.scheduler.get_all_posts(status="scheduled")
        
        upcoming = [
            p for p in all_posts
            if p.get("scheduled_time") and
            datetime.fromisoformat(p.get("scheduled_time")) > now
        ]
        
        # Ordenar por fecha
        upcoming.sort(key=lambda x: x.get("scheduled_time", ""))
        
        return upcoming[:limit]
    
    def get_recent_activity(
        self,
        manager: Any,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Obtener actividad reciente
        
        Args:
            manager: Instancia de CommunityManager
            limit: Límite de items
            
        Returns:
            Lista de actividades
        """
        activities = []
        
        # Posts publicados recientemente
        published_posts = manager.scheduler.get_all_posts(status="published")
        published_posts.sort(
            key=lambda x: x.get("published_at", ""),
            reverse=True
        )
        
        for post in published_posts[:limit // 2]:
            activities.append({
                "type": "post_published",
                "timestamp": post.get("published_at"),
                "data": {
                    "post_id": post.get("id"),
                    "platforms": post.get("platforms", [])
                }
            })
        
        # Notificaciones recientes (si hay notification service)
        if hasattr(manager, 'notification_service'):
            notifications = manager.notification_service.get_notifications(limit=limit // 2)
            for notif in notifications:
                activities.append({
                    "type": notif.get("type"),
                    "timestamp": notif.get("timestamp"),
                    "data": notif.get("data", {})
                })
        
        # Ordenar por timestamp
        activities.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return activities[:limit]
    
    def get_performance_metrics(
        self,
        analytics_service: Any,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtener métricas de performance
        
        Args:
            analytics_service: Instancia de AnalyticsService
            days: Días hacia atrás
            
        Returns:
            Dict con métricas de performance
        """
        summary = analytics_service.get_summary(days)
        
        best_posts = analytics_service.get_best_performing_posts(limit=5)
        
        return {
            "period_days": days,
            "total_engagement": summary.get("total_engagement_across_platforms", 0),
            "total_posts": summary.get("total_posts_across_platforms", 0),
            "average_engagement_rate": summary.get("average_engagement_rate_across_platforms", 0),
            "best_posts": best_posts,
            "platform_breakdown": summary.get("platform_breakdown", {})
        }
    
    def get_content_calendar_summary(
        self,
        manager: Any,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtener resumen del calendario de contenido
        
        Args:
            manager: Instancia de CommunityManager
            days: Días hacia adelante
            
        Returns:
            Dict con resumen del calendario
        """
        from datetime import datetime, timedelta
        
        end_date = datetime.now() + timedelta(days=days)
        all_posts = manager.scheduler.get_all_posts()
        
        scheduled = [
            p for p in all_posts
            if p.get("scheduled_time") and
            datetime.fromisoformat(p.get("scheduled_time")) <= end_date
        ]
        
        by_day = defaultdict(int)
        by_platform = defaultdict(int)
        
        for post in scheduled:
            scheduled_time = datetime.fromisoformat(post.get("scheduled_time"))
            day_key = scheduled_time.date().isoformat()
            by_day[day_key] += 1
            
            for platform in post.get("platforms", []):
                by_platform[platform] += 1
        
        return {
            "period_days": days,
            "total_scheduled": len(scheduled),
            "by_day": dict(by_day),
            "by_platform": dict(by_platform),
            "upcoming_count": len([p for p in scheduled if datetime.fromisoformat(p.get("scheduled_time")) > datetime.now()])
        }
    
    def get_platform_health(
        self,
        manager: Any
    ) -> Dict[str, Any]:
        """
        Obtener estado de salud de las plataformas
        
        Args:
            manager: Instancia de CommunityManager
            
        Returns:
            Dict con estado de salud
        """
        connected = manager.social_connector.get_connected_platforms()
        connections = manager.social_connector.connections
        
        health_status = {}
        
        for platform in connected:
            conn_info = connections.get(platform, {})
            health_status[platform] = {
                "connected": conn_info.get("connected", False),
                "connected_at": conn_info.get("connected_at"),
                "status": "healthy" if conn_info.get("connected") else "disconnected"
            }
        
        return {
            "total_platforms": len(connected),
            "connected_count": len([p for p in connected if connections.get(p, {}).get("connected")]),
            "platforms": health_status
        }



