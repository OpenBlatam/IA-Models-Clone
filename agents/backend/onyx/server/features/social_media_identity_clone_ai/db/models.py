"""
Modelos de base de datos para Social Media Identity Clone AI
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime, 
    JSON, ForeignKey, Index, Boolean
)
from sqlalchemy.orm import relationship

from .base import Base


class IdentityProfileModel(Base):
    """Modelo de base de datos para IdentityProfile"""
    __tablename__ = "identity_profiles"
    
    id = Column(String(64), primary_key=True, index=True)
    username = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    
    # Estadísticas
    total_videos = Column(Integer, default=0)
    total_posts = Column(Integer, default=0)
    total_comments = Column(Integer, default=0)
    
    # Knowledge base
    knowledge_base = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    social_profiles = relationship("SocialProfileModel", back_populates="identity_profile", cascade="all, delete-orphan")
    content_analysis = relationship("ContentAnalysisModel", back_populates="identity_profile", uselist=False, cascade="all, delete-orphan")
    generated_content = relationship("GeneratedContentModel", back_populates="identity_profile", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_username", "username"),
        Index("idx_created_at", "created_at"),
    )


class SocialProfileModel(Base):
    """Modelo de base de datos para SocialProfile"""
    __tablename__ = "social_profiles"
    
    id = Column(String(64), primary_key=True, index=True)
    identity_profile_id = Column(String(64), ForeignKey("identity_profiles.id"), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)  # tiktok, instagram, youtube
    username = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    profile_image_url = Column(String(500), nullable=True)
    
    # Estadísticas
    followers_count = Column(Integer, nullable=True)
    following_count = Column(Integer, nullable=True)
    posts_count = Column(Integer, nullable=True)
    
    # Timestamps
    extracted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Metadata y contenido (serializado)
    metadata = Column(JSON, nullable=True)
    videos_data = Column(JSON, nullable=True)  # Lista de videos serializados
    posts_data = Column(JSON, nullable=True)  # Lista de posts serializados
    comments_data = Column(JSON, nullable=True)  # Lista de comentarios serializados
    
    # Relationship
    identity_profile = relationship("IdentityProfileModel", back_populates="social_profiles")
    
    __table_args__ = (
        Index("idx_identity_platform", "identity_profile_id", "platform"),
        Index("idx_username_platform", "username", "platform"),
    )


class ContentAnalysisModel(Base):
    """Modelo de base de datos para ContentAnalysis"""
    __tablename__ = "content_analyses"
    
    id = Column(String(64), primary_key=True, index=True)
    identity_profile_id = Column(String(64), ForeignKey("identity_profiles.id"), nullable=False, unique=True, index=True)
    
    # Análisis
    topics = Column(JSON, nullable=True)  # Lista de topics
    themes = Column(JSON, nullable=True)  # Lista de themes
    tone = Column(String(100), nullable=True)
    personality_traits = Column(JSON, nullable=True)
    communication_style = Column(String(100), nullable=True)
    common_phrases = Column(JSON, nullable=True)
    values = Column(JSON, nullable=True)
    interests = Column(JSON, nullable=True)
    language_patterns = Column(JSON, nullable=True)
    sentiment_analysis = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship
    identity_profile = relationship("IdentityProfileModel", back_populates="content_analysis")


class GeneratedContentModel(Base):
    """Modelo de base de datos para GeneratedContent"""
    __tablename__ = "generated_content"
    
    id = Column(String(64), primary_key=True, index=True)
    identity_profile_id = Column(String(64), ForeignKey("identity_profiles.id"), nullable=False, index=True)
    
    # Contenido
    platform = Column(String(50), nullable=False, index=True)
    content_type = Column(String(50), nullable=False)  # video, post, story, reel, comment
    content = Column(Text, nullable=False)
    title = Column(String(500), nullable=True)
    hashtags = Column(JSON, nullable=True)  # Lista de hashtags
    
    # Calidad
    confidence_score = Column(Float, nullable=True)
    
    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationship
    identity_profile = relationship("IdentityProfileModel", back_populates="generated_content")
    
    __table_args__ = (
        Index("idx_identity_platform", "identity_profile_id", "platform"),
        Index("idx_generated_at", "generated_at"),
    )




