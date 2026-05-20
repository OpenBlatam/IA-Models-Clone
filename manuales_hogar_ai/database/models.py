"""
Modelos de Base de Datos
========================

Modelos SQLAlchemy para el sistema de manuales.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()


class Manual(Base):
    """Modelo para almacenar manuales generados."""
    
    __tablename__ = "manuales"
    
    id = Column(Integer, primary_key=True, index=True)
    problem_description = Column(Text, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    manual_content = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, default=0)
    image_analysis = Column(Text, nullable=True)
    detected_category = Column(String(50), nullable=True)
    images_count = Column(Integer, default=0)
    format = Column(String(20), default="lego")
    
    # Nuevos campos mejorados
    title = Column(String(200), nullable=True, index=True)  # Título del manual
    difficulty = Column(String(20), nullable=True)  # Fácil, Media, Difícil
    estimated_time = Column(String(50), nullable=True)  # Tiempo estimado
    tools_required = Column(Text, nullable=True)  # Lista de herramientas
    materials_required = Column(Text, nullable=True)  # Lista de materiales
    safety_warnings = Column(Text, nullable=True)  # Advertencias de seguridad
    
    # Métricas de uso
    view_count = Column(Integer, default=0)
    favorite_count = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)  # Nuevo: contador de compartidos
    
    # Metadata adicional
    user_id = Column(String(100), nullable=True, index=True)  # ID de usuario (si hay auth)
    is_public = Column(Boolean, default=True)  # Si es público o privado
    tags = Column(Text, nullable=True)  # Tags separados por comas
    share_token = Column(String(64), unique=True, nullable=True, index=True)  # Token para compartir
    
    # Versionado
    version = Column(Integer, default=1)
    parent_manual_id = Column(Integer, ForeignKey('manuales.id'), nullable=True)  # Manual padre si es versión
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relaciones
    versions = relationship("Manual", backref="parent_manual", remote_side=[id])
    
    # Índices compuestos para búsquedas
    __table_args__ = (
        Index('idx_category_created', 'category', 'created_at'),
        Index('idx_problem_description', 'problem_description', postgresql_using='gin'),
        Index('idx_difficulty', 'difficulty'),
        Index('idx_rating', 'average_rating'),
        Index('idx_public', 'is_public', 'created_at'),
        Index('idx_share_token', 'share_token'),
    )


class ManualCache(Base):
    """Modelo para cache persistente de manuales."""
    
    __tablename__ = "manuales_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(64), unique=True, nullable=False, index=True)
    problem_description_hash = Column(String(64), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    manual_content = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, default=0)
    
    # Metadata de cache
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('idx_expires_at', 'expires_at'),
    )


class UsageStats(Base):
    """Modelo para estadísticas de uso."""
    
    __tablename__ = "manuales_usage_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    model_used = Column(String(100), nullable=True)
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    avg_tokens_per_request = Column(Float, default=0.0)
    total_images_processed = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    __table_args__ = (
        Index('idx_date_category', 'date', 'category'),
        Index('idx_date', 'date'),
    )


class ManualRating(Base):
    """Modelo para ratings de manuales."""
    
    __tablename__ = "manuales_ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(Integer, nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    __table_args__ = (
        Index('idx_manual_user', 'manual_id', 'user_id', unique=True),
        Index('idx_rating', 'rating'),
    )


class ManualFavorite(Base):
    """Modelo para favoritos de manuales."""
    
    __tablename__ = "manuales_favorites"
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(Integer, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_manual_user_fav', 'manual_id', 'user_id', unique=True),
        Index('idx_user_favorites', 'user_id', 'created_at'),
    )


class ManualShare(Base):
    """Modelo para compartir manuales."""
    
    __tablename__ = "manuales_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(Integer, nullable=False, index=True)
    share_token = Column(String(64), unique=True, nullable=False, index=True)
    shared_by = Column(String(100), nullable=True)  # Usuario que compartió
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Expiración opcional
    access_count = Column(Integer, default=0)  # Veces accedido
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('idx_share_token', 'share_token'),
        Index('idx_manual_share', 'manual_id'),
    )


class ManualTemplate(Base):
    """Modelo para plantillas de manuales."""
    
    __tablename__ = "manuales_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False, index=True)
    template_content = Column(Text, nullable=False)  # Contenido de la plantilla
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    __table_args__ = (
        Index('idx_template_category', 'category'),
        Index('idx_template_public', 'is_public'),
    )


class Notification(Base):
    """Modelo para notificaciones."""
    
    __tablename__ = "manuales_notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    manual_id = Column(Integer, nullable=True, index=True)
    type = Column(String(50), nullable=False)  # rating, favorite, comment, etc.
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_user_unread', 'user_id', 'is_read'),
        Index('idx_user_created', 'user_id', 'created_at'),
    )
