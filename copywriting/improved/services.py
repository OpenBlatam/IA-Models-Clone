"""
Copywriting Services
===================

Clean, async service layer following FastAPI best practices.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta

import httpx
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Float, Integer, JSON

from .schemas import (
    CopywritingRequest,
    CopywritingResponse,
    CopywritingVariant,
    FeedbackRequest,
    FeedbackResponse,
    BatchCopywritingRequest,
    BatchCopywritingResponse,
    HealthCheckResponse
)
from .exceptions import (
    ContentGenerationError,
    ExternalServiceError,
    DatabaseError,
    ValidationError
)
from .config import get_settings, get_database_settings, get_redis_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for database models"""
    pass


class CopywritingRecord(Base):
    """Database model for copywriting records"""
    __tablename__ = "copywriting_records"
    
    id: Mapped[UUID] = mapped_column(String, primary_key=True, default=uuid4)
    request_id: Mapped[UUID] = mapped_column(String, nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    target_audience: Mapped[str] = mapped_column(String(200), nullable=False)
    tone: Mapped[str] = mapped_column(String(50), nullable=False)
    style: Mapped[str] = mapped_column(String(50), nullable=False)
    purpose: Mapped[str] = mapped_column(String(50), nullable=False)
    variants: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeedbackRecord(Base):
    """Database model for feedback records"""
    __tablename__ = "feedback_records"
    
    id: Mapped[UUID] = mapped_column(String, primary_key=True, default=uuid4)
    variant_id: Mapped[UUID] = mapped_column(String, nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    feedback_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    improvements: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    is_helpful: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CopywritingService:
    """Main copywriting service with async operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_settings = get_database_settings()
        self.redis_settings = get_redis_settings()
        self._engine = None
        self._session_factory = None
        self._redis = None
        self._http_client = None
        self._startup_time = time.time()
    
    async def startup(self) -> None:
        """Initialize service dependencies"""
        try:
            # Initialize database
            self._engine = create_async_engine(
                self.db_settings.url,
                pool_size=self.db_settings.pool_size,
                max_overflow=self.db_settings.max_overflow,
                echo=self.db_settings.echo
            )
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize Redis
            self._redis = await aioredis.from_url(
                self.redis_settings.url,
                max_connections=self.redis_settings.max_connections,
                socket_timeout=self.redis_settings.socket_timeout,
                socket_connect_timeout=self.redis_settings.socket_connect_timeout
            )
            
            # Initialize HTTP client
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            
            logger.info("CopywritingService started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start CopywritingService: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cleanup service dependencies"""
        try:
            if self._http_client:
                await self._http_client.aclose()
            
            if self._redis:
                await self._redis.close()
            
            if self._engine:
                await self._engine.dispose()
            
            logger.info("CopywritingService shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during CopywritingService shutdown: {e}")
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self._session_factory:
            raise DatabaseError("Database not initialized")
        return self._session_factory()
    
    async def generate_copywriting(self, request: CopywritingRequest) -> CopywritingResponse:
        """Generate copywriting content"""
        start_time = time.time()
        request_id = uuid4()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self._get_from_cache(cache_key)
            if cached_response:
                logger.info(f"Cache hit for request {request_id}")
                return cached_response
            
            # Generate content
            variants = await self._generate_variants(request)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create response
            response = CopywritingResponse(
                request_id=request_id,
                variants=variants,
                metadata={
                    "tone": request.tone,
                    "style": request.style,
                    "purpose": request.purpose,
                    "language": request.language,
                    "creativity_level": request.creativity_level
                },
                processing_time_ms=processing_time_ms
            )
            
            # Cache response
            await self._set_cache(cache_key, response)
            
            # Store in database
            await self._store_copywriting_record(request, response)
            
            logger.info(f"Generated copywriting for request {request_id} in {processing_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate copywriting for request {request_id}: {e}")
            raise ContentGenerationError(
                message="Failed to generate copywriting content",
                details={"request_id": str(request_id), "error": str(e)},
                request_id=request_id
            )
    
    async def generate_batch_copywriting(self, request: BatchCopywritingRequest) -> BatchCopywritingResponse:
        """Generate copywriting content in batch"""
        start_time = time.time()
        results = []
        failed_requests = []
        
        # Process requests concurrently
        tasks = []
        for i, copy_request in enumerate(request.requests):
            task = self._process_single_request(copy_request, i)
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                failed_requests.append({
                    "index": i,
                    "error": str(response),
                    "request": request.requests[i].model_dump()
                })
            else:
                results.append(response)
        
        total_processing_time_ms = int((time.time() - start_time) * 1000)
        
        return BatchCopywritingResponse(
            batch_id=request.batch_id,
            results=results,
            failed_requests=failed_requests,
            total_processing_time_ms=total_processing_time_ms
        )
    
    async def submit_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Submit feedback for a copywriting variant"""
        try:
            # Store feedback in database
            async with await self.get_session() as session:
                feedback_record = FeedbackRecord(
                    variant_id=request.variant_id,
                    rating=request.rating,
                    feedback_text=request.feedback_text,
                    improvements=request.improvements,
                    is_helpful=request.is_helpful
                )
                session.add(feedback_record)
                await session.commit()
            
            logger.info(f"Feedback submitted for variant {request.variant_id}")
            
            return FeedbackResponse(
                variant_id=request.variant_id,
                status="received",
                message="Feedback received successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to submit feedback for variant {request.variant_id}: {e}")
            raise DatabaseError(
                message="Failed to submit feedback",
                details={"variant_id": str(request.variant_id), "error": str(e)}
            )
    
    async def get_health_status(self) -> HealthCheckResponse:
        """Get health status of the service"""
        try:
            # Check database connection
            async with await self.get_session() as session:
                await session.execute("SELECT 1")
            db_status = "healthy"
        except Exception:
            db_status = "unhealthy"
        
        try:
            # Check Redis connection
            await self._redis.ping()
            redis_status = "healthy"
        except Exception:
            redis_status = "unhealthy"
        
        # Determine overall status
        if db_status == "healthy" and redis_status == "healthy":
            status = "healthy"
        elif db_status == "unhealthy" or redis_status == "unhealthy":
            status = "degraded"
        else:
            status = "unhealthy"
        
        uptime_seconds = time.time() - self._startup_time
        
        return HealthCheckResponse(
            status=status,
            uptime_seconds=uptime_seconds,
            dependencies={
                "database": db_status,
                "redis": redis_status
            }
        )
    
    async def _process_single_request(self, request: CopywritingRequest, index: int) -> CopywritingResponse:
        """Process a single copywriting request"""
        try:
            return await self.generate_copywriting(request)
        except Exception as e:
            logger.error(f"Failed to process request at index {index}: {e}")
            raise
    
    async def _generate_variants(self, request: CopywritingRequest) -> List[CopywritingVariant]:
        """Generate copywriting variants"""
        # This is a simplified implementation
        # In a real application, this would call an AI service or use templates
        
        variants = []
        for i in range(request.variants_count):
            # Generate content based on request parameters
            content = await self._generate_content(request, i)
            
            variant = CopywritingVariant(
                title=f"{request.topic} - Variant {i + 1}",
                content=content,
                word_count=len(content.split()),
                cta="Learn More" if request.include_cta else None,
                confidence_score=0.8 + (i * 0.05)  # Simulate confidence scores
            )
            variants.append(variant)
        
        return variants
    
    async def _generate_content(self, request: CopywritingRequest, variant_index: int) -> str:
        """Generate content for a specific variant"""
        # This is a placeholder implementation
        # In a real application, this would integrate with an AI service
        
        base_content = f"""
        {request.topic}
        
        For {request.target_audience}, this content addresses your needs with a {request.tone} tone.
        
        Key points:
        {chr(10).join(f"• {point}" for point in request.key_points)}
        
        This {request.style} approach is designed for {request.purpose}.
        """
        
        if request.include_cta:
            base_content += "\n\nTake action now and experience the difference!"
        
        return base_content.strip()
    
    def _generate_cache_key(self, request: CopywritingRequest) -> str:
        """Generate cache key for request"""
        # Create a hash of the request for caching
        request_dict = request.model_dump()
        # Remove timestamp-dependent fields
        request_dict.pop("created_at", None)
        return f"copywriting:{hash(str(sorted(request_dict.items())))}"
    
    async def _get_from_cache(self, key: str) -> Optional[CopywritingResponse]:
        """Get response from cache"""
        if not self.settings.cache.enabled or not self._redis:
            return None
        
        try:
            cached_data = await self._redis.get(key)
            if cached_data:
                # Deserialize from JSON
                import json
                data = json.loads(cached_data)
                return CopywritingResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to get from cache: {e}")
        
        return None
    
    async def _set_cache(self, key: str, response: CopywritingResponse) -> None:
        """Set response in cache"""
        if not self.settings.cache.enabled or not self._redis:
            return
        
        try:
            # Serialize to JSON
            data = response.model_dump_json()
            await self._redis.setex(
                key,
                self.settings.cache.copywriting_ttl,
                data
            )
        except Exception as e:
            logger.warning(f"Failed to set cache: {e}")
    
    async def _store_copywriting_record(self, request: CopywritingRequest, response: CopywritingResponse) -> None:
        """Store copywriting record in database"""
        try:
            async with await self.get_session() as session:
                record = CopywritingRecord(
                    request_id=response.request_id,
                    topic=request.topic,
                    target_audience=request.target_audience,
                    tone=request.tone,
                    style=request.style,
                    purpose=request.purpose,
                    variants=[variant.model_dump() for variant in response.variants],
                    processing_time_ms=response.processing_time_ms
                )
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning(f"Failed to store copywriting record: {e}")


# Global service instance
_service_instance: Optional[CopywritingService] = None


async def get_copywriting_service() -> CopywritingService:
    """Get copywriting service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = CopywritingService()
        await _service_instance.startup()
    return _service_instance


async def cleanup_copywriting_service() -> None:
    """Cleanup copywriting service"""
    global _service_instance
    if _service_instance:
        await _service_instance.shutdown()
        _service_instance = None






























