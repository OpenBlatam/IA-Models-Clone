from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from domain.entities import (
from domain.interfaces import CopywritingRepository
from typing import Any, List, Dict, Optional
"""
Infrastructure Repositories
===========================

Repository implementations for data persistence.
"""



    CopywritingRequest,
    CopywritingResponse,
    PerformanceMetrics,
    RequestStatus,
    CopywritingStyle,
    CopywritingTone
)

logger = logging.getLogger(__name__)

Base = declarative_base()


class CopywritingRequestModel(Base):
    """SQLAlchemy model for copywriting requests."""
    
    __tablename__ = "copywriting_requests"
    
    id = Column(String, primary_key=True)
    prompt = Column(Text, nullable=False)
    style = Column(String, nullable=False)
    tone = Column(String, nullable=False)
    length = Column(Integer, nullable=False)
    creativity = Column(Float, nullable=False)
    language = Column(String, nullable=False)
    target_audience = Column(String, nullable=True)
    keywords = Column(JSON, nullable=False, default=list)
    metadata = Column(JSON, nullable=False, default=dict)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class CopywritingResponseModel(Base):
    """SQLAlchemy model for copywriting responses."""
    
    __tablename__ = "copywriting_responses"
    
    id = Column(String, primary_key=True)
    request_id = Column(String, nullable=False)
    generated_text = Column(Text, nullable=False)
    processing_time = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    suggestions = Column(JSON, nullable=False, default=list)
    metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.now)


class PerformanceMetricsModel(Base):
    """SQLAlchemy model for performance metrics."""
    
    __tablename__ = "performance_metrics"
    
    id = Column(String, primary_key=True)
    request_count = Column(Integer, nullable=False, default=0)
    average_processing_time = Column(Float, nullable=False, default=0.0)
    cache_hit_ratio = Column(Float, nullable=False, default=0.0)
    error_rate = Column(Float, nullable=False, default=0.0)
    system_metrics = Column(JSON, nullable=False, default=dict)
    ai_metrics = Column(JSON, nullable=False, default=dict)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)


class PostgresCopywritingRepository(CopywritingRepository):
    """PostgreSQL implementation of copywriting repository."""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        
    """__init__ function."""
self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self) -> Any:
        """Initialize database connection and create tables."""
        if self._initialized:
            return
        
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                echo=False
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info("PostgreSQL repository initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL repository: {e}")
            raise
    
    async def cleanup(self) -> Any:
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("PostgreSQL repository cleaned up")
    
    async async def save_request(self, request: CopywritingRequest) -> CopywritingRequest:
        """Save a copywriting request."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                # Convert domain entity to model
                model = CopywritingRequestModel(
                    id=request.id,
                    prompt=request.prompt,
                    style=request.style.value,
                    tone=request.tone.value,
                    length=request.length,
                    creativity=request.creativity,
                    language=request.language,
                    target_audience=request.target_audience,
                    keywords=request.keywords,
                    metadata=request.metadata,
                    status=request.status.value,
                    created_at=request.created_at,
                    updated_at=request.updated_at
                )
                
                session.add(model)
                await session.commit()
                
                logger.debug(f"Saved request {request.id}")
                return request
                
        except Exception as e:
            logger.error(f"Error saving request {request.id}: {e}")
            raise
    
    async def save_response(self, response: CopywritingResponse) -> CopywritingResponse:
        """Save a copywriting response."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                # Convert domain entity to model
                model = CopywritingResponseModel(
                    id=response.id,
                    request_id=response.request_id,
                    generated_text=response.generated_text,
                    processing_time=response.processing_time,
                    model_used=response.model_used,
                    confidence_score=response.confidence_score,
                    suggestions=response.suggestions,
                    metadata=response.metadata,
                    created_at=response.created_at
                )
                
                session.add(model)
                await session.commit()
                
                logger.debug(f"Saved response {response.id}")
                return response
                
        except Exception as e:
            logger.error(f"Error saving response {response.id}: {e}")
            raise
    
    async async def get_request_by_id(self, request_id: str) -> Optional[CopywritingRequest]:
        """Get request by ID."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM copywriting_requests WHERE id = '{request_id}'"
                )
                row = result.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_request(row)
                
        except Exception as e:
            logger.error(f"Error getting request {request_id}: {e}")
            raise
    
    async def get_response_by_id(self, response_id: str) -> Optional[CopywritingResponse]:
        """Get response by ID."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM copywriting_responses WHERE id = '{response_id}'"
                )
                row = result.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_response(row)
                
        except Exception as e:
            logger.error(f"Error getting response {response_id}: {e}")
            raise
    
    async async def get_responses_by_request_id(self, request_id: str) -> List[CopywritingResponse]:
        """Get all responses for a request."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM copywriting_responses WHERE request_id = '{request_id}' ORDER BY created_at DESC"
                )
                rows = result.fetchall()
                
                return [self._row_to_response(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting responses for request {request_id}: {e}")
            raise
    
    async async def update_request_status(self, request_id: str, status: RequestStatus) -> bool:
        """Update request status."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    f"UPDATE copywriting_requests SET status = '{status.value}', updated_at = NOW() WHERE id = '{request_id}'"
                )
                await session.commit()
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating status for request {request_id}: {e}")
            raise
    
    async def get_user_history(self, user_id: str, limit: int = 50) -> List[CopywritingRequest]:
        """Get user's copywriting history."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM copywriting_requests WHERE metadata->>'user_id' = '{user_id}' ORDER BY created_at DESC LIMIT {limit}"
                )
                rows = result.fetchall()
                
                return [self._row_to_request(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting history for user {user_id}: {e}")
            raise
    
    async async def get_requests_by_status(self, status: RequestStatus) -> List[CopywritingRequest]:
        """Get requests by status."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM copywriting_requests WHERE status = '{status.value}' ORDER BY created_at DESC"
                )
                rows = result.fetchall()
                
                return [self._row_to_request(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting requests by status {status.value}: {e}")
            raise
    
    async async def delete_request(self, request_id: str) -> bool:
        """Delete a request."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                # Delete associated responses first
                await session.execute(
                    f"DELETE FROM copywriting_responses WHERE request_id = '{request_id}'"
                )
                
                # Delete request
                result = await session.execute(
                    f"DELETE FROM copywriting_requests WHERE id = '{request_id}'"
                )
                await session.commit()
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error deleting request {request_id}: {e}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized")
        
        try:
            async with self.session_factory() as session:
                # Total requests
                result = await session.execute("SELECT COUNT(*) FROM copywriting_requests")
                total_requests = result.fetchone()[0]
                
                # Requests by status
                result = await session.execute(
                    "SELECT status, COUNT(*) FROM copywriting_requests GROUP BY status"
                )
                requests_by_status = dict(result.fetchall())
                
                # Total responses
                result = await session.execute("SELECT COUNT(*) FROM copywriting_responses")
                total_responses = result.fetchone()[0]
                
                # Average processing time
                result = await session.execute(
                    "SELECT AVG(processing_time) FROM copywriting_responses"
                )
                avg_processing_time = result.fetchone()[0] or 0.0
                
                # Recent activity (last 24 hours)
                result = await session.execute(
                    "SELECT COUNT(*) FROM copywriting_requests WHERE created_at >= NOW() - INTERVAL '24 hours'"
                )
                recent_requests = result.fetchone()[0]
                
                return {
                    "total_requests": total_requests,
                    "total_responses": total_responses,
                    "requests_by_status": requests_by_status,
                    "average_processing_time": avg_processing_time,
                    "recent_requests_24h": recent_requests,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise
    
    async def _row_to_request(self, row) -> CopywritingRequest:
        """Convert database row to CopywritingRequest entity."""
        return CopywritingRequest(
            id=row.id,
            prompt=row.prompt,
            style=CopywritingStyle(row.style),
            tone=CopywritingTone(row.tone),
            length=row.length,
            creativity=row.creativity,
            language=row.language,
            target_audience=row.target_audience,
            keywords=row.keywords or [],
            metadata=row.metadata or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
            status=RequestStatus(row.status)
        )
    
    def _row_to_response(self, row) -> CopywritingResponse:
        """Convert database row to CopywritingResponse entity."""
        return CopywritingResponse(
            id=row.id,
            request_id=row.request_id,
            generated_text=row.generated_text,
            processing_time=row.processing_time,
            model_used=row.model_used,
            confidence_score=row.confidence_score,
            suggestions=row.suggestions or [],
            metadata=row.metadata or {},
            created_at=row.created_at
        ) 