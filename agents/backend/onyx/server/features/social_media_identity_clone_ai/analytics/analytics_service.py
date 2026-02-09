"""
Servicio de analytics para análisis de uso y rendimiento
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..db.models import IdentityProfileModel, GeneratedContentModel, SocialProfileModel
from ..db.base import get_db_session
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Servicio para análisis de datos y métricas"""
    
    def __init__(self):
        self.metrics = get_metrics_collector()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        with get_db_session() as db:
            total_identities = db.query(func.count(IdentityProfileModel.id)).scalar() or 0
            total_generated_content = db.query(func.count(GeneratedContentModel.id)).scalar() or 0
            total_social_profiles = db.query(func.count(SocialProfileModel.id)).scalar() or 0
            
            # Contenido generado por plataforma
            content_by_platform = (
                db.query(
                    GeneratedContentModel.platform,
                    func.count(GeneratedContentModel.id).label("count")
                )
                .group_by(GeneratedContentModel.platform)
                .all()
            )
            
            # Identidades creadas por día (últimos 7 días)
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_identities = (
                db.query(func.count(IdentityProfileModel.id))
                .filter(IdentityProfileModel.created_at >= seven_days_ago)
                .scalar() or 0
            )
            
            return {
                "total_identities": total_identities,
                "total_generated_content": total_generated_content,
                "total_social_profiles": total_social_profiles,
                "recent_identities_7d": recent_identities,
                "content_by_platform": {
                    platform: count for platform, count in content_by_platform
                },
                "metrics": self.metrics.get_all_metrics()
            }
    
    def get_identity_analytics(self, identity_id: str) -> Dict[str, Any]:
        """Obtiene analytics de una identidad específica"""
        with get_db_session() as db:
            identity = db.query(IdentityProfileModel).filter_by(id=identity_id).first()
            if not identity:
                return {}
            
            # Contenido generado para esta identidad
            content_count = (
                db.query(func.count(GeneratedContentModel.id))
                .filter_by(identity_profile_id=identity_id)
                .scalar() or 0
            )
            
            # Contenido por plataforma
            content_by_platform = (
                db.query(
                    GeneratedContentModel.platform,
                    func.count(GeneratedContentModel.id).label("count")
                )
                .filter_by(identity_profile_id=identity_id)
                .group_by(GeneratedContentModel.platform)
                .all()
            )
            
            # Último contenido generado
            last_content = (
                db.query(GeneratedContentModel)
                .filter_by(identity_profile_id=identity_id)
                .order_by(GeneratedContentModel.generated_at.desc())
                .first()
            )
            
            return {
                "identity_id": identity_id,
                "username": identity.username,
                "created_at": identity.created_at.isoformat(),
                "total_videos": identity.total_videos,
                "total_posts": identity.total_posts,
                "total_comments": identity.total_comments,
                "generated_content_count": content_count,
                "content_by_platform": {
                    platform: count for platform, count in content_by_platform
                },
                "last_content_generated": last_content.generated_at.isoformat() if last_content else None,
            }
    
    def get_usage_trends(self, days: int = 30) -> Dict[str, Any]:
        """Obtiene tendencias de uso"""
        with get_db_session() as db:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Identidades creadas por día
            identities_by_day = (
                db.query(
                    func.date(IdentityProfileModel.created_at).label("date"),
                    func.count(IdentityProfileModel.id).label("count")
                )
                .filter(IdentityProfileModel.created_at >= start_date)
                .group_by(func.date(IdentityProfileModel.created_at))
                .all()
            )
            
            # Contenido generado por día
            content_by_day = (
                db.query(
                    func.date(GeneratedContentModel.generated_at).label("date"),
                    func.count(GeneratedContentModel.id).label("count")
                )
                .filter(GeneratedContentModel.generated_at >= start_date)
                .group_by(func.date(GeneratedContentModel.generated_at))
                .all()
            )
            
            return {
                "period_days": days,
                "identities_by_day": {
                    str(date): count for date, count in identities_by_day
                },
                "content_by_day": {
                    str(date): count for date, count in content_by_day
                }
            }




