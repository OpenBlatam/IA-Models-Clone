"""
PDF Variantes Database Models and Migrations
Modelos de base de datos y migraciones para el sistema PDF Variantes
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    """Modelo de usuario"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relaciones
    documents = relationship("Document", back_populates="owner")
    collaborations = relationship("Collaboration", back_populates="user")

class Document(Base):
    """Modelo de documento PDF"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), default="application/pdf")
    status = Column(String(50), default="uploaded")
    page_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    language = Column(String(10), default="en")
    metadata = Column(JSON)
    content_hash = Column(String(64), unique=True)
    ipfs_hash = Column(String(64))
    blockchain_tx_hash = Column(String(66))
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    owner = relationship("User", back_populates="documents")
    variants = relationship("Variant", back_populates="document")
    topics = relationship("Topic", back_populates="document")
    brainstorm_ideas = relationship("BrainstormIdea", back_populates="document")
    collaborations = relationship("Collaboration", back_populates="document")

class Variant(Base):
    """Modelo de variante de documento"""
    __tablename__ = "variants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    configuration = Column(JSON)
    status = Column(String(50), default="generated")
    similarity_score = Column(Float, default=0.0)
    creativity_score = Column(Float, default=0.0)
    quality_score = Column(Float, default=0.0)
    generation_time = Column(Float, default=0.0)
    model_used = Column(String(100))
    differences = Column(JSON)
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    document = relationship("Document", back_populates="variants")

class Topic(Base):
    """Modelo de tema extraído"""
    __tablename__ = "topics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    topic = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False)
    relevance_score = Column(Float, default=0.0)
    mentions = Column(Integer, default=0)
    context = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    document = relationship("Document", back_populates="topics")

class BrainstormIdea(Base):
    """Modelo de idea de brainstorming"""
    __tablename__ = "brainstorm_ideas"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    idea = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    related_topics = Column(JSON)
    potential_impact = Column(String(50), default="medium")
    implementation_difficulty = Column(String(50), default="medium")
    priority_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    document = relationship("Document", back_populates="brainstorm_ideas")

class Collaboration(Base):
    """Modelo de colaboración"""
    __tablename__ = "collaborations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(String(50), default="viewer")
    permissions = Column(JSON)
    invited_at = Column(DateTime, default=datetime.utcnow)
    accepted_at = Column(DateTime)
    last_accessed = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relaciones
    document = relationship("Document", back_populates="collaborations")
    user = relationship("User", back_populates="collaborations")

class Annotation(Base):
    """Modelo de anotación"""
    __tablename__ = "annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    x_position = Column(Float)
    y_position = Column(Float)
    width = Column(Float)
    height = Column(Float)
    content = Column(Text, nullable=False)
    annotation_type = Column(String(50), default="comment")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Export(Base):
    """Modelo de exportación"""
    __tablename__ = "exports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    export_format = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    status = Column(String(50), default="completed")
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

class Analytics(Base):
    """Modelo de analytics"""
    __tablename__ = "analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    event_type = Column(String(100), nullable=False)
    event_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(String(500))

class DatabaseManager:
    """Gestor de base de datos"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Inicializar conexión a base de datos"""
        try:
            # Crear motor asíncrono
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            # Crear factory de sesiones
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Database Manager: {e}")
            raise
    
    async def create_tables(self):
        """Crear tablas en la base de datos"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Obtener sesión de base de datos"""
        return self.session_factory()
    
    async def close(self):
        """Cerrar conexión a base de datos"""
        try:
            if self.engine:
                await self.engine.dispose()
            
            logger.info("Database Manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Database Manager: {e}")

# Factory function
async def create_database_manager(database_url: str) -> DatabaseManager:
    """Crear gestor de base de datos"""
    manager = DatabaseManager(database_url)
    await manager.initialize()
    return manager
