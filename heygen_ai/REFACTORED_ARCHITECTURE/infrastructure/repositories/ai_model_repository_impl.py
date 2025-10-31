#!/usr/bin/env python3
"""
🗄️ HeyGen AI - AI Model Repository Implementation (Infrastructure Layer)
========================================================================

This module implements the AI Model repository using SQLAlchemy and PostgreSQL,
following the Repository pattern and Clean Architecture principles.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
import logging
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ...domain.entities.ai_model import AIModel, ModelType, ModelStatus, OptimizationLevel
from ...domain.repositories.base_repository import (
    BaseRepository, RepositoryQuery, RepositoryResult, RepositoryFilter,
    RepositoryError, EntityNotFoundError, DuplicateEntityError
)

logger = logging.getLogger(__name__)

# SQLAlchemy models
Base = declarative_base()

class AIModelModel(Base):
    """SQLAlchemy model for AI Model"""
    __tablename__ = 'ai_models'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)
    model_status = Column(String(50), nullable=False, default='training')
    current_version = Column(String(20), nullable=False, default='1.0.0')
    description = Column(Text)
    configuration = Column(JSON, nullable=False)
    metrics = Column(JSON)
    model_path = Column(String(500))
    deployment_url = Column(String(500))
    training_history = Column(JSON, default=list)
    deployment_history = Column(JSON, default=list)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String(20), nullable=False, default='active')
    metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    created_by = Column(String(100))
    updated_by = Column(String(100))

class AIModelRepositoryImpl(BaseRepository[AIModel]):
    """
    AI Model repository implementation using SQLAlchemy and PostgreSQL.
    
    This implementation provides concrete database operations for AI models
    following the Repository pattern.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize repository with database connection.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
    
    def _get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def _model_to_entity(self, model: AIModelModel) -> AIModel:
        """Convert SQLAlchemy model to domain entity"""
        # Parse configuration
        config_data = model.configuration
        configuration = ModelConfiguration(
            model_type=ModelType(config_data['model_type']),
            architecture=config_data['architecture'],
            hyperparameters=config_data['hyperparameters'],
            optimization_level=OptimizationLevel(config_data['optimization_level']),
            input_shape=config_data['input_shape'],
            output_shape=config_data['output_shape'],
            batch_size=config_data.get('batch_size', 32),
            learning_rate=config_data.get('learning_rate', 0.001),
            epochs=config_data.get('epochs', 100),
            dropout_rate=config_data.get('dropout_rate', 0.1)
        )
        
        # Parse metrics if available
        metrics = None
        if model.metrics:
            metrics = ModelMetrics(**model.metrics)
        
        # Parse version
        version_parts = model.current_version.split('.')
        current_version = ModelVersion(
            major=int(version_parts[0]),
            minor=int(version_parts[1]),
            patch=int(version_parts[2])
        )
        
        # Create entity
        entity = AIModel(
            id=model.id,
            name=model.name,
            model_type=ModelType(model.model_type),
            configuration=configuration,
            description=model.description or "",
            created_at=model.created_at,
            updated_at=model.updated_at,
            version=model.version,
            status=model.status,
            metadata=model.metadata or {},
            tags=model.tags or [],
            created_by=model.created_by,
            updated_by=model.updated_by
        )
        
        # Set additional properties
        entity.model_status = ModelStatus(model.model_status)
        entity.current_version = current_version
        entity.metrics = metrics
        entity.model_path = model.model_path
        entity.deployment_url = model.deployment_url
        entity.training_history = model.training_history or []
        entity.deployment_history = model.deployment_history or []
        
        return entity
    
    def _entity_to_model(self, entity: AIModel) -> AIModelModel:
        """Convert domain entity to SQLAlchemy model"""
        return AIModelModel(
            id=entity.id,
            name=entity.name,
            model_type=entity.model_type.value,
            model_status=entity.model_status.value,
            current_version=str(entity.current_version),
            description=entity.description,
            configuration={
                'model_type': entity.configuration.model_type.value,
                'architecture': entity.configuration.architecture,
                'hyperparameters': entity.configuration.hyperparameters,
                'optimization_level': entity.configuration.optimization_level.value,
                'input_shape': entity.configuration.input_shape,
                'output_shape': entity.configuration.output_shape,
                'batch_size': entity.configuration.batch_size,
                'learning_rate': entity.configuration.learning_rate,
                'epochs': entity.configuration.epochs,
                'dropout_rate': entity.configuration.dropout_rate
            },
            metrics=entity.metrics.__dict__ if entity.metrics else None,
            model_path=entity.model_path,
            deployment_url=entity.deployment_url,
            training_history=entity.training_history,
            deployment_history=entity.deployment_history,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            version=entity.version,
            status=entity.status.value,
            metadata=entity.metadata,
            tags=entity.tags,
            created_by=entity.created_by,
            updated_by=entity.updated_by
        )
    
    async def create(self, entity: AIModel) -> AIModel:
        """Create a new AI model"""
        try:
            with self._get_session() as session:
                # Check for duplicates
                existing = session.query(AIModelModel).filter_by(name=entity.name).first()
                if existing:
                    raise DuplicateEntityError(f"Model with name '{entity.name}' already exists")
                
                # Convert to model
                model = self._entity_to_model(entity)
                
                # Save to database
                session.add(model)
                session.commit()
                session.refresh(model)
                
                # Convert back to entity
                created_entity = self._model_to_entity(model)
                
                logger.info(f"Created AI model: {entity.name} (ID: {entity.id})")
                return created_entity
                
        except DuplicateEntityError:
            raise
        except Exception as e:
            logger.error(f"Failed to create AI model: {e}")
            raise RepositoryError(f"Failed to create AI model: {str(e)}", e)
    
    async def get_by_id(self, entity_id: str) -> Optional[AIModel]:
        """Get AI model by ID"""
        try:
            with self._get_session() as session:
                model = session.query(AIModelModel).filter_by(id=entity_id).first()
                if not model:
                    return None
                
                return self._model_to_entity(model)
                
        except Exception as e:
            logger.error(f"Failed to get AI model by ID {entity_id}: {e}")
            raise RepositoryError(f"Failed to get AI model: {str(e)}", e)
    
    async def update(self, entity: AIModel) -> AIModel:
        """Update AI model"""
        try:
            with self._get_session() as session:
                # Check if entity exists
                existing = session.query(AIModelModel).filter_by(id=entity.id).first()
                if not existing:
                    raise EntityNotFoundError(f"AI model with ID {entity.id} not found")
                
                # Update fields
                existing.name = entity.name
                existing.model_type = entity.model_type.value
                existing.model_status = entity.model_status.value
                existing.current_version = str(entity.current_version)
                existing.description = entity.description
                existing.configuration = {
                    'model_type': entity.configuration.model_type.value,
                    'architecture': entity.configuration.architecture,
                    'hyperparameters': entity.configuration.hyperparameters,
                    'optimization_level': entity.configuration.optimization_level.value,
                    'input_shape': entity.configuration.input_shape,
                    'output_shape': entity.configuration.output_shape,
                    'batch_size': entity.configuration.batch_size,
                    'learning_rate': entity.configuration.learning_rate,
                    'epochs': entity.configuration.epochs,
                    'dropout_rate': entity.configuration.dropout_rate
                }
                existing.metrics = entity.metrics.__dict__ if entity.metrics else None
                existing.model_path = entity.model_path
                existing.deployment_url = entity.deployment_url
                existing.training_history = entity.training_history
                existing.deployment_history = entity.deployment_history
                existing.updated_at = entity.updated_at
                existing.version = entity.version
                existing.status = entity.status.value
                existing.metadata = entity.metadata
                existing.tags = entity.tags
                existing.updated_by = entity.updated_by
                
                # Save changes
                session.commit()
                session.refresh(existing)
                
                # Convert back to entity
                updated_entity = self._model_to_entity(existing)
                
                logger.info(f"Updated AI model: {entity.name} (ID: {entity.id})")
                return updated_entity
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update AI model {entity.id}: {e}")
            raise RepositoryError(f"Failed to update AI model: {str(e)}", e)
    
    async def delete(self, entity_id: str) -> bool:
        """Delete AI model"""
        try:
            with self._get_session() as session:
                model = session.query(AIModelModel).filter_by(id=entity_id).first()
                if not model:
                    return False
                
                session.delete(model)
                session.commit()
                
                logger.info(f"Deleted AI model: {model.name} (ID: {entity_id})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete AI model {entity_id}: {e}")
            raise RepositoryError(f"Failed to delete AI model: {str(e)}", e)
    
    async def find(self, query: RepositoryQuery) -> RepositoryResult[AIModel]:
        """Find AI models matching query"""
        try:
            with self._get_session() as session:
                # Start with base query
                db_query = session.query(AIModelModel)
                
                # Apply filters
                for filter_obj in query.filters:
                    if filter_obj.operator == 'eq':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) == filter_obj.value)
                    elif filter_obj.operator == 'ne':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) != filter_obj.value)
                    elif filter_obj.operator == 'gt':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) > filter_obj.value)
                    elif filter_obj.operator == 'gte':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) >= filter_obj.value)
                    elif filter_obj.operator == 'lt':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) < filter_obj.value)
                    elif filter_obj.operator == 'lte':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) <= filter_obj.value)
                    elif filter_obj.operator == 'in':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field).in_(filter_obj.value))
                    elif filter_obj.operator == 'nin':
                        db_query = db_query.filter(~getattr(AIModelModel, filter_obj.field).in_(filter_obj.value))
                    elif filter_obj.operator == 'like':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field).like(f"%{filter_obj.value}%"))
                    elif filter_obj.operator == 'ilike':
                        db_query = db_query.filter(getattr(AIModelModel, filter_obj.field).ilike(f"%{filter_obj.value}%"))
                
                # Apply search
                if query.search:
                    search_filter = (
                        AIModelModel.name.ilike(f"%{query.search}%") |
                        AIModelModel.description.ilike(f"%{query.search}%")
                    )
                    db_query = db_query.filter(search_filter)
                
                # Get total count
                total_count = db_query.count()
                
                # Apply sorting
                for sort_obj in query.sorts:
                    field = getattr(AIModelModel, sort_obj.field)
                    if sort_obj.direction == 'desc':
                        field = field.desc()
                    db_query = db_query.order_by(field)
                
                # Apply pagination
                if query.pagination:
                    offset = query.pagination.offset
                    limit = query.pagination.page_size
                    db_query = db_query.offset(offset).limit(limit)
                    
                    page = query.pagination.page
                    page_size = query.pagination.page_size
                else:
                    page = 1
                    page_size = total_count
                
                # Execute query
                models = db_query.all()
                
                # Convert to entities
                entities = [self._model_to_entity(model) for model in models]
                
                # Calculate pagination info
                total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0
                has_next = page < total_pages
                has_previous = page > 1
                
                return RepositoryResult(
                    entities=entities,
                    total_count=total_count,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages,
                    has_next=has_next,
                    has_previous=has_previous
                )
                
        except Exception as e:
            logger.error(f"Failed to find AI models: {e}")
            raise RepositoryError(f"Failed to find AI models: {str(e)}", e)
    
    async def count(self, query: Optional[RepositoryQuery] = None) -> int:
        """Count AI models matching query"""
        try:
            with self._get_session() as session:
                db_query = session.query(AIModelModel)
                
                if query:
                    # Apply filters (same logic as find method)
                    for filter_obj in query.filters:
                        if filter_obj.operator == 'eq':
                            db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) == filter_obj.value)
                        elif filter_obj.operator == 'ne':
                            db_query = db_query.filter(getattr(AIModelModel, filter_obj.field) != filter_obj.value)
                        # Add other operators as needed
                
                return db_query.count()
                
        except Exception as e:
            logger.error(f"Failed to count AI models: {e}")
            raise RepositoryError(f"Failed to count AI models: {str(e)}", e)
    
    async def exists(self, entity_id: str) -> bool:
        """Check if AI model exists"""
        try:
            with self._get_session() as session:
                count = session.query(AIModelModel).filter_by(id=entity_id).count()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to check if AI model exists {entity_id}: {e}")
            raise RepositoryError(f"Failed to check if AI model exists: {str(e)}", e)


