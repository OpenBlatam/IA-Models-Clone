"""
Database Models and Operations for Document Workflow Chain
=========================================================

This module provides database models, operations, and persistence layer
for the Document Workflow Chain service using SQLAlchemy.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Database base class
Base = declarative_base()

class WorkflowChainModel(Base):
    """Database model for workflow chains"""
    __tablename__ = "workflow_chains"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    root_node_id = Column(UUID(as_uuid=True), nullable=True)
    status = Column(String(50), default="active")  # active, paused, completed, error
    settings = Column(JSON)
    user_id = Column(String(255), nullable=True)  # For multi-user support
    metadata = Column(JSON)
    
    # Relationships
    nodes = relationship("DocumentNodeModel", back_populates="chain", cascade="all, delete-orphan")

class DocumentNodeModel(Base):
    """Database model for document nodes"""
    __tablename__ = "document_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chain_id = Column(UUID(as_uuid=True), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    parent_id = Column(UUID(as_uuid=True), nullable=True)
    children_ids = Column(JSON)  # List of child node IDs
    metadata = Column(JSON)
    word_count = Column(Integer, default=0)
    ai_model_used = Column(String(100))
    tokens_used = Column(Integer, default=0)
    generation_time = Column(Float, default=0.0)
    quality_score = Column(Float, nullable=True)
    
    # Relationships
    chain = relationship("WorkflowChainModel", back_populates="nodes")

class WorkflowStatsModel(Base):
    """Database model for workflow statistics"""
    __tablename__ = "workflow_stats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chain_id = Column(UUID(as_uuid=True), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    total_documents = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    avg_generation_time = Column(Float, default=0.0)
    avg_quality_score = Column(Float, default=0.0)
    metadata = Column(JSON)

class AIClientStatsModel(Base):
    """Database model for AI client statistics"""
    __tablename__ = "ai_client_stats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_type = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_errors = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)
    metadata = Column(JSON)

class DatabaseManager:
    """Database manager for workflow chain operations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.async_session = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            # Create async session factory
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    async def create_workflow_chain(
        self,
        name: str,
        description: str,
        root_node_id: Optional[str] = None,
        status: str = "active",
        settings: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowChainModel:
        """Create a new workflow chain in database"""
        async with self.async_session() as session:
            try:
                chain = WorkflowChainModel(
                    name=name,
                    description=description,
                    root_node_id=root_node_id,
                    status=status,
                    settings=settings or {},
                    user_id=user_id,
                    metadata=metadata or {}
                )
                
                session.add(chain)
                await session.commit()
                await session.refresh(chain)
                
                logger.info(f"Created workflow chain in database: {chain.id}")
                return chain
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to create workflow chain: {str(e)}")
                raise
    
    async def get_workflow_chain(self, chain_id: str) -> Optional[WorkflowChainModel]:
        """Get workflow chain by ID"""
        async with self.async_session() as session:
            try:
                result = await session.get(WorkflowChainModel, chain_id)
                return result
            except Exception as e:
                logger.error(f"Failed to get workflow chain {chain_id}: {str(e)}")
                return None
    
    async def update_workflow_chain(
        self,
        chain_id: str,
        **updates
    ) -> Optional[WorkflowChainModel]:
        """Update workflow chain"""
        async with self.async_session() as session:
            try:
                chain = await session.get(WorkflowChainModel, chain_id)
                if not chain:
                    return None
                
                for key, value in updates.items():
                    if hasattr(chain, key):
                        setattr(chain, key, value)
                
                chain.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(chain)
                
                logger.info(f"Updated workflow chain: {chain_id}")
                return chain
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update workflow chain {chain_id}: {str(e)}")
                raise
    
    async def delete_workflow_chain(self, chain_id: str) -> bool:
        """Delete workflow chain and all its nodes"""
        async with self.async_session() as session:
            try:
                chain = await session.get(WorkflowChainModel, chain_id)
                if not chain:
                    return False
                
                await session.delete(chain)
                await session.commit()
                
                logger.info(f"Deleted workflow chain: {chain_id}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete workflow chain {chain_id}: {str(e)}")
                raise
    
    async def create_document_node(
        self,
        chain_id: str,
        title: str,
        content: str,
        prompt: str,
        parent_id: Optional[str] = None,
        children_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ai_model_used: Optional[str] = None,
        tokens_used: int = 0,
        generation_time: float = 0.0,
        quality_score: Optional[float] = None
    ) -> DocumentNodeModel:
        """Create a new document node"""
        async with self.async_session() as session:
            try:
                node = DocumentNodeModel(
                    chain_id=chain_id,
                    title=title,
                    content=content,
                    prompt=prompt,
                    parent_id=parent_id,
                    children_ids=children_ids or [],
                    metadata=metadata or {},
                    word_count=len(content.split()),
                    ai_model_used=ai_model_used,
                    tokens_used=tokens_used,
                    generation_time=generation_time,
                    quality_score=quality_score
                )
                
                session.add(node)
                await session.commit()
                await session.refresh(node)
                
                logger.info(f"Created document node: {node.id}")
                return node
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to create document node: {str(e)}")
                raise
    
    async def get_document_node(self, node_id: str) -> Optional[DocumentNodeModel]:
        """Get document node by ID"""
        async with self.async_session() as session:
            try:
                result = await session.get(DocumentNodeModel, node_id)
                return result
            except Exception as e:
                logger.error(f"Failed to get document node {node_id}: {str(e)}")
                return None
    
    async def get_chain_nodes(self, chain_id: str) -> List[DocumentNodeModel]:
        """Get all nodes for a workflow chain"""
        async with self.async_session() as session:
            try:
                from sqlalchemy import select
                stmt = select(DocumentNodeModel).where(
                    DocumentNodeModel.chain_id == chain_id
                ).order_by(DocumentNodeModel.generated_at)
                
                result = await session.execute(stmt)
                return result.scalars().all()
                
            except Exception as e:
                logger.error(f"Failed to get chain nodes for {chain_id}: {str(e)}")
                return []
    
    async def update_document_node(
        self,
        node_id: str,
        **updates
    ) -> Optional[DocumentNodeModel]:
        """Update document node"""
        async with self.async_session() as session:
            try:
                node = await session.get(DocumentNodeModel, node_id)
                if not node:
                    return None
                
                for key, value in updates.items():
                    if hasattr(node, key):
                        setattr(node, key, value)
                
                await session.commit()
                await session.refresh(node)
                
                logger.info(f"Updated document node: {node_id}")
                return node
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update document node {node_id}: {str(e)}")
                raise
    
    async def get_user_workflows(self, user_id: str) -> List[WorkflowChainModel]:
        """Get all workflows for a user"""
        async with self.async_session() as session:
            try:
                from sqlalchemy import select
                stmt = select(WorkflowChainModel).where(
                    WorkflowChainModel.user_id == user_id
                ).order_by(WorkflowChainModel.created_at.desc())
                
                result = await session.execute(stmt)
                return result.scalars().all()
                
            except Exception as e:
                logger.error(f"Failed to get user workflows for {user_id}: {str(e)}")
                return []
    
    async def get_workflow_stats(self, chain_id: str) -> Dict[str, Any]:
        """Get statistics for a workflow chain"""
        async with self.async_session() as session:
            try:
                from sqlalchemy import select, func
                
                # Get node count and stats
                stmt = select(
                    func.count(DocumentNodeModel.id).label('total_documents'),
                    func.sum(DocumentNodeModel.word_count).label('total_words'),
                    func.sum(DocumentNodeModel.tokens_used).label('total_tokens'),
                    func.avg(DocumentNodeModel.generation_time).label('avg_generation_time'),
                    func.avg(DocumentNodeModel.quality_score).label('avg_quality_score')
                ).where(DocumentNodeModel.chain_id == chain_id)
                
                result = await session.execute(stmt)
                stats = result.first()
                
                return {
                    "total_documents": stats.total_documents or 0,
                    "total_words": stats.total_words or 0,
                    "total_tokens": stats.total_tokens or 0,
                    "avg_generation_time": float(stats.avg_generation_time or 0),
                    "avg_quality_score": float(stats.avg_quality_score or 0)
                }
                
            except Exception as e:
                logger.error(f"Failed to get workflow stats for {chain_id}: {str(e)}")
                return {}
    
    async def save_ai_client_stats(
        self,
        client_type: str,
        model: str,
        requests: int,
        tokens: int,
        errors: int,
        avg_response_time: float,
        success_rate: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save AI client statistics"""
        async with self.async_session() as session:
            try:
                stats = AIClientStatsModel(
                    client_type=client_type,
                    model=model,
                    total_requests=requests,
                    total_tokens=tokens,
                    total_errors=errors,
                    avg_response_time=avg_response_time,
                    success_rate=success_rate,
                    metadata=metadata or {}
                )
                
                session.add(stats)
                await session.commit()
                
                logger.info(f"Saved AI client stats for {client_type}/{model}")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to save AI client stats: {str(e)}")
                raise
    
    async def search_workflows(
        self,
        query: str,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[WorkflowChainModel]:
        """Search workflows by name or description"""
        async with self.async_session() as session:
            try:
                from sqlalchemy import select, or_
                
                stmt = select(WorkflowChainModel).where(
                    or_(
                        WorkflowChainModel.name.ilike(f"%{query}%"),
                        WorkflowChainModel.description.ilike(f"%{query}%")
                    )
                )
                
                if user_id:
                    stmt = stmt.where(WorkflowChainModel.user_id == user_id)
                
                if status:
                    stmt = stmt.where(WorkflowChainModel.status == status)
                
                stmt = stmt.order_by(WorkflowChainModel.created_at.desc()).limit(limit)
                
                result = await session.execute(stmt)
                return result.scalars().all()
                
            except Exception as e:
                logger.error(f"Failed to search workflows: {str(e)}")
                return []

# Database utility functions
async def create_database_manager(database_url: str) -> DatabaseManager:
    """Create and initialize database manager"""
    manager = DatabaseManager(database_url)
    await manager.initialize()
    return manager

# Example usage and testing
if __name__ == "__main__":
    async def test_database():
        """Test database functionality"""
        print("🧪 Testing Database Operations")
        print("=" * 40)
        
        # Use in-memory SQLite for testing
        database_url = "sqlite+aiosqlite:///:memory:"
        
        try:
            # Create database manager
            db_manager = await create_database_manager(database_url)
            
            # Test workflow chain creation
            chain = await db_manager.create_workflow_chain(
                name="Test Workflow",
                description="Test workflow for database testing",
                user_id="test_user"
            )
            print(f"✅ Created workflow chain: {chain.id}")
            
            # Test document node creation
            node = await db_manager.create_document_node(
                chain_id=str(chain.id),
                title="Test Document",
                content="This is a test document content.",
                prompt="Generate a test document",
                ai_model_used="test-model",
                tokens_used=100,
                generation_time=1.5
            )
            print(f"✅ Created document node: {node.id}")
            
            # Test retrieval
            retrieved_chain = await db_manager.get_workflow_chain(str(chain.id))
            print(f"✅ Retrieved workflow chain: {retrieved_chain.name}")
            
            # Test stats
            stats = await db_manager.get_workflow_stats(str(chain.id))
            print(f"✅ Workflow stats: {stats}")
            
            # Cleanup
            await db_manager.close()
            print("✅ Database test completed successfully")
            
        except Exception as e:
            print(f"❌ Database test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run test
    asyncio.run(test_database())


