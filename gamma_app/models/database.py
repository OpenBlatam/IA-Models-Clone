"""
Gamma App - Database Models
SQLAlchemy models for the application
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, JSON, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID
import json

Base = declarative_base()

class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(String(20), default="user")  # user, admin, editor
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    preferences = Column(JSON, default=dict)
    
    # Relationships
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    content_items = relationship("ContentItem", back_populates="creator", cascade="all, delete-orphan")
    collaboration_sessions = relationship("CollaborationSession", back_populates="owner", cascade="all, delete-orphan")

class Project(Base):
    """Project model"""
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    content_items = relationship("ContentItem", back_populates="project", cascade="all, delete-orphan")
    collaboration_sessions = relationship("CollaborationSession", back_populates="project", cascade="all, delete-orphan")

class ContentItem(Base):
    """Content item model"""
    __tablename__ = "content_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    content_type = Column(String(50), nullable=False)  # presentation, document, web_page
    content_data = Column(JSON, nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default="draft")  # draft, published, archived
    version = Column(Integer, default=1)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime)
    
    # Relationships
    project = relationship("Project", back_populates="content_items")
    creator = relationship("User", back_populates="content_items")
    exports = relationship("Export", back_populates="content_item", cascade="all, delete-orphan")
    collaboration_sessions = relationship("CollaborationSession", back_populates="content_item", cascade="all, delete-orphan")

class Export(Base):
    """Export model"""
    __tablename__ = "exports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_item_id = Column(UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False)
    export_format = Column(String(20), nullable=False)  # pdf, pptx, html, etc.
    file_path = Column(String(500))
    file_size = Column(Integer)
    export_config = Column(JSON, default=dict)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    content_item = relationship("ContentItem", back_populates="exports")

class CollaborationSession(Base):
    """Collaboration session model"""
    __tablename__ = "collaboration_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    content_item_id = Column(UUID(as_uuid=True), ForeignKey("content_items.id"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="collaboration_sessions")
    content_item = relationship("ContentItem", back_populates="collaboration_sessions")
    owner = relationship("User", back_populates="collaboration_sessions")
    participants = relationship("CollaborationParticipant", back_populates="session", cascade="all, delete-orphan")
    events = relationship("CollaborationEvent", back_populates="session", cascade="all, delete-orphan")

class CollaborationParticipant(Base):
    """Collaboration participant model"""
    __tablename__ = "collaboration_participants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaboration_sessions.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(String(20), default="editor")  # owner, editor, viewer, commentor
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    cursor_position = Column(JSON)
    is_online = Column(Boolean, default=False)
    
    # Relationships
    session = relationship("CollaborationSession", back_populates="participants")
    user = relationship("User")

class CollaborationEvent(Base):
    """Collaboration event model"""
    __tablename__ = "collaboration_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaboration_sessions.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    event_type = Column(String(50), nullable=False)  # cursor_update, content_edit, etc.
    event_data = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    version = Column(Integer, default=1)
    
    # Relationships
    session = relationship("CollaborationSession", back_populates="events")
    user = relationship("User")

class AnalyticsEvent(Base):
    """Analytics event model"""
    __tablename__ = "analytics_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    session_id = Column(String(100))
    event_data = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Relationships
    user = relationship("User")

class PerformanceMetric(Base):
    """Performance metric model"""
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)
    tags = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Index for efficient querying
    __table_args__ = (
        {"postgresql_partition_by": "RANGE (timestamp)"},
    )

class SecurityEvent(Base):
    """Security event model"""
    __tablename__ = "security_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)
    source_ip = Column(String(45), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    description = Column(Text, nullable=False)
    metadata = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    resolved = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User")

class CacheEntry(Base):
    """Cache entry model"""
    __tablename__ = "cache_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(255), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)
    namespace = Column(String(100), default="default", index=True)
    expires_at = Column(DateTime, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)

class AIModel(Base):
    """AI Model metadata model"""
    __tablename__ = "ai_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)
    model_size = Column(String(20), nullable=False)
    model_path = Column(String(500), nullable=False)
    tokenizer_path = Column(String(500), nullable=False)
    config = Column(JSON, default=dict)
    is_loaded = Column(Boolean, default=False)
    is_fine_tuned = Column(Boolean, default=False)
    performance_metrics = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Template(Base):
    """Template model"""
    __tablename__ = "templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    template_type = Column(String(50), nullable=False)  # presentation, document, web_page
    category = Column(String(50))
    description = Column(Text)
    template_data = Column(JSON, nullable=False)
    preview_image = Column(String(500))
    is_public = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creator = relationship("User")

# Database connection and session management
class DatabaseManager:
    """Database manager for connection and session handling"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()

# Global database manager instance
db_manager = None

def init_database(database_url: str):
    """Initialize database connection"""
    global db_manager
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager

def get_db():
    """Get database session dependency"""
    if db_manager is None:
        raise RuntimeError("Database not initialized")
    
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


























