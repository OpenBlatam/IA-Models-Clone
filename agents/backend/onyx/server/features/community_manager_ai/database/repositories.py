"""
Repositories - Repositorios de Datos
====================================

Repositorios para acceso a datos con SQLAlchemy.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from .models import Post, Meme, Template, PlatformConnection, AnalyticsMetric, Notification

logger = logging.getLogger(__name__)


class PostRepository:
    """Repositorio para Posts"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, post_data: Dict[str, Any]) -> Post:
        """Crear un nuevo post"""
        post = Post(**post_data)
        self.db.add(post)
        self.db.commit()
        self.db.refresh(post)
        return post
    
    def get_by_id(self, post_id: str) -> Optional[Post]:
        """Obtener post por ID"""
        return self.db.query(Post).filter(Post.id == post_id).first()
    
    def get_all(
        self,
        status: Optional[str] = None,
        platform: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Post]:
        """Obtener todos los posts con filtros"""
        query = self.db.query(Post)
        
        if status:
            query = query.filter(Post.status == status)
        
        if platform:
            query = query.filter(Post.platforms.contains([platform]))
        
        query = query.order_by(desc(Post.created_at))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_pending(self, before_time: Optional[datetime] = None) -> List[Post]:
        """Obtener posts pendientes"""
        query = self.db.query(Post).filter(
            and_(
                Post.status == "scheduled",
                Post.scheduled_time <= (before_time or datetime.now())
            )
        )
        return query.order_by(Post.scheduled_time).all()
    
    def update(self, post_id: str, update_data: Dict[str, Any]) -> Optional[Post]:
        """Actualizar un post"""
        post = self.get_by_id(post_id)
        if not post:
            return None
        
        for key, value in update_data.items():
            setattr(post, key, value)
        
        post.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(post)
        return post
    
    def delete(self, post_id: str) -> bool:
        """Eliminar un post"""
        post = self.get_by_id(post_id)
        if not post:
            return False
        
        self.db.delete(post)
        self.db.commit()
        return True


class MemeRepository:
    """Repositorio para Memes"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, meme_data: Dict[str, Any]) -> Meme:
        """Crear un nuevo meme"""
        meme = Meme(**meme_data)
        self.db.add(meme)
        self.db.commit()
        self.db.refresh(meme)
        return meme
    
    def get_by_id(self, meme_id: str) -> Optional[Meme]:
        """Obtener meme por ID"""
        return self.db.query(Meme).filter(Meme.id == meme_id).first()
    
    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Meme]:
        """Buscar memes"""
        db_query = self.db.query(Meme)
        
        if category:
            db_query = db_query.filter(Meme.category == category)
        
        if tags:
            for tag in tags:
                db_query = db_query.filter(Meme.tags.contains([tag]))
        
        if query:
            db_query = db_query.filter(
                or_(
                    Meme.caption.contains(query),
                    Meme.tags.contains([query])
                )
            )
        
        return db_query.all()
    
    def get_random(self, category: Optional[str] = None) -> Optional[Meme]:
        """Obtener meme aleatorio"""
        import random
        
        query = self.db.query(Meme)
        if category:
            query = query.filter(Meme.category == category)
        
        memes = query.all()
        if not memes:
            return None
        
        meme = random.choice(memes)
        meme.usage_count += 1
        meme.updated_at = datetime.utcnow()
        self.db.commit()
        return meme


class AnalyticsRepository:
    """Repositorio para Analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, metric_data: Dict[str, Any]) -> AnalyticsMetric:
        """Crear métrica"""
        metric = AnalyticsMetric(**metric_data)
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        return metric
    
    def get_by_post(self, post_id: str, platform: Optional[str] = None) -> List[AnalyticsMetric]:
        """Obtener métricas de un post"""
        query = self.db.query(AnalyticsMetric).filter(AnalyticsMetric.post_id == post_id)
        
        if platform:
            query = query.filter(AnalyticsMetric.platform == platform)
        
        return query.order_by(desc(AnalyticsMetric.recorded_at)).all()
    
    def get_platform_stats(
        self,
        platform: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Obtener estadísticas de plataforma"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.db.query(AnalyticsMetric).filter(
            and_(
                AnalyticsMetric.platform == platform,
                AnalyticsMetric.recorded_at >= cutoff_date
            )
        ).all()
        
        if not metrics:
            return {
                "platform": platform,
                "total_posts": 0,
                "total_engagement": 0,
                "average_engagement_rate": 0.0
            }
        
        total_likes = sum(m.likes for m in metrics)
        total_comments = sum(m.comments for m in metrics)
        total_shares = sum(m.shares + m.retweets for m in metrics)
        total_reach = sum(m.reach + m.impressions for m in metrics)
        total_engagement = total_likes + total_comments + total_shares
        
        avg_engagement_rate = (
            sum(m.engagement_rate for m in metrics) / len(metrics)
            if metrics else 0.0
        )
        
        return {
            "platform": platform,
            "period_days": days,
            "total_posts": len(metrics),
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_shares": total_shares,
            "total_reach": total_reach,
            "total_engagement": total_engagement,
            "average_engagement_rate": round(avg_engagement_rate, 2)
        }


class TemplateRepository:
    """Repositorio para Templates"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, template_data: Dict[str, Any]) -> Template:
        """Crear plantilla"""
        template = Template(**template_data)
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        return template
    
    def get_by_id(self, template_id: str) -> Optional[Template]:
        """Obtener plantilla por ID"""
        return self.db.query(Template).filter(Template.id == template_id).first()
    
    def search(
        self,
        query: Optional[str] = None,
        platform: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Template]:
        """Buscar plantillas"""
        db_query = self.db.query(Template)
        
        if platform:
            db_query = db_query.filter(Template.platform == platform)
        
        if category:
            db_query = db_query.filter(Template.category == category)
        
        if query:
            db_query = db_query.filter(
                or_(
                    Template.name.contains(query),
                    Template.content.contains(query)
                )
            )
        
        return db_query.all()




