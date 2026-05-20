"""
Database Models - Modelos de Base de Datos
===========================================

Modelos SQLAlchemy para persistencia de datos.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Post(Base):
    """Modelo de Post"""
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    platforms = Column(JSON, nullable=False)  # Lista de plataformas
    scheduled_time = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    status = Column(String, default="scheduled")  # scheduled, published, cancelled, failed
    media_paths = Column(JSON, default=list)  # Lista de rutas
    tags = Column(JSON, default=list)  # Lista de tags
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    analytics = relationship("AnalyticsMetric", back_populates="post", cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )
    
    def to_dict(self):
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "content": self.content,
            "platforms": self.platforms,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "status": self.status,
            "media_paths": self.media_paths,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Meme(Base):
    """Modelo de Meme"""
    __tablename__ = "memes"
    
    id = Column(String, primary_key=True)
    image_path = Column(String, nullable=False)
    original_path = Column(String, nullable=True)
    caption = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    category = Column(String, default="general")
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "original_path": self.original_path,
            "caption": self.caption,
            "tags": self.tags,
            "category": self.category,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Template(Base):
    """Modelo de Plantilla"""
    __tablename__ = "templates"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    platform = Column(String, nullable=True)
    variables = Column(JSON, default=list)
    category = Column(String, default="general")
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "platform": self.platform,
            "variables": self.variables,
            "category": self.category,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PlatformConnection(Base):
    """Modelo de Conexión a Plataforma"""
    __tablename__ = "platform_connections"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String, nullable=False, unique=True)
    connected = Column(Boolean, default=False)
    credentials = Column(JSON, nullable=True)  # Credenciales encriptadas
    connected_at = Column(DateTime, nullable=True)
    last_sync_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "platform": self.platform,
            "connected": self.connected,
            "credentials": {k: "***" for k in (self.credentials or {}).keys()},  # Ocultar credenciales
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AnalyticsMetric(Base):
    """Modelo de Métrica de Analytics"""
    __tablename__ = "analytics_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String, ForeignKey("posts.id"), nullable=False)
    platform = Column(String, nullable=False)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    retweets = Column(Integer, default=0)  # Para Twitter
    reach = Column(Integer, default=0)
    impressions = Column(Integer, default=0)
    views = Column(Integer, default=0)  # Para videos
    engagement_rate = Column(Float, default=0.0)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    post = relationship("Post", back_populates="analytics")
    
    def to_dict(self):
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "post_id": self.post_id,
            "platform": self.platform,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "retweets": self.retweets,
            "reach": self.reach,
            "impressions": self.impressions,
            "views": self.views,
            "engagement_rate": self.engagement_rate,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Notification(Base):
    """Modelo de Notificación"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSON, default=dict)
    read = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "type": self.type,
            "message": self.message,
            "data": self.data,
            "read": self.read,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

