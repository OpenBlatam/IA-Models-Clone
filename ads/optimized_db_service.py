from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime, timedelta
import asyncio
import json
import hashlib
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import and_, desc, func, select, update, delete
from sqlalchemy.orm import selectinload
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from functools import lru_cache
from onyx.db.ads import AdsGeneration, BackgroundRemoval, AdsAnalytics
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.config import settings
from typing import Any, List, Dict, Optional
import logging
"""
Optimized database service for ads functionality with connection pooling and caching.
"""


logger = setup_logger()

class OptimizedAdsDBService:
    """Optimized service for handling ads-related database operations with connection pooling and caching."""
    
    def __init__(self) -> Any:
        """Initialize with connection pooling and caching."""
        self._engine = None
        self._session_factory = None
        self._redis_client = None
        self._connection_pool = None
        self._cache_ttl = 3600  # 1 hour default
        
    @property
    async def engine(self) -> Any:
        """Lazy initialization of async database engine with connection pooling."""
        if self._engine is None:
            self._engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_timeout=30
            )
        return self._engine
    
    @property
    async def session_factory(self) -> Any:
        """Lazy initialization of session factory."""
        if self._session_factory is None:
            engine = await self.engine
            self._session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        return self._session_factory
    
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client for caching."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50
            )
        return self._redis_client
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with proper error handling."""
        session_factory = await self.session_factory
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_data = json.dumps(kwargs, sort_keys=True)
        return f"ads_db:{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis."""
        try:
            redis = await self.redis_client
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _set_cached_result(self, cache_key: str, result: Dict[str, Any], ttl: int = None):
        """Set result in cache with TTL."""
        try:
            redis = await self.redis_client
            ttl = ttl or self._cache_ttl
            await redis.setex(cache_key, ttl, json.dumps(result, default=str))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        try:
            redis = await self.redis_client
            keys = await redis.keys(pattern)
            if keys:
                await redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    async def create_ads_generation(
        self,
        user_id: int,
        url: str,
        type: str,
        content: Dict[str, Any],
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        brand_voice: Optional[Dict[str, Any]] = None,
        audience_profile: Optional[Dict[str, Any]] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> AdsGeneration:
        """Create a new ads generation record with caching."""
        async with self.get_session() as session:
            ads_generation = AdsGeneration(
                user_id=user_id,
                url=url,
                type=type,
                prompt=prompt,
                content=content,
                metadata=metadata,
                brand_voice=brand_voice,
                audience_profile=audience_profile,
                project_context=project_context
            )
            session.add(ads_generation)
            await session.flush()
            await session.refresh(ads_generation)
            
            # Invalidate user's ads cache
            await self._invalidate_cache_pattern(f"ads_db:list:*:user_id:{user_id}")
            
            return ads_generation

    async def get_ads_generation(
        self,
        user_id: int,
        ads_id: int
    ) -> Optional[AdsGeneration]:
        """Get an ads generation record by ID with caching."""
        cache_key = await self._get_cache_key("get_ads", user_id=user_id, ads_id=ads_id)
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            return AdsGeneration(**cached_result)
        
        async with self.get_session() as session:
            result = await session.execute(
                select(AdsGeneration).filter(
                    and_(
                        AdsGeneration.id == ads_id,
                        AdsGeneration.user_id == user_id,
                        AdsGeneration.is_deleted == False
                    )
                )
            )
            ads_generation = result.scalar_one_or_none()
            
            if ads_generation:
                # Cache the result
                await self._set_cached_result(cache_key, ads_generation.__dict__)
            
            return ads_generation

    async def list_ads_generations(
        self,
        user_id: int,
        type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AdsGeneration]:
        """List ads generation records for a user with caching and pagination."""
        cache_key = await self._get_cache_key(
            "list_ads", 
            user_id=user_id, 
            type=type, 
            limit=limit, 
            offset=offset
        )
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            return [AdsGeneration(**item) for item in cached_result]
        
        async with self.get_session() as session:
            query = select(AdsGeneration).filter(
                and_(
                    AdsGeneration.user_id == user_id,
                    AdsGeneration.is_deleted == False
                )
            )
            
            if type:
                query = query.filter(AdsGeneration.type == type)
            
            query = query.order_by(desc(AdsGeneration.created_at)).offset(offset).limit(limit)
            result = await session.execute(query)
            ads_generations = result.scalars().all()
            
            # Cache the result
            cache_data = [ads.__dict__ for ads in ads_generations]
            await self._set_cached_result(cache_key, cache_data, ttl=1800)  # 30 minutes
            
            return ads_generations

    async def create_background_removal(
        self,
        user_id: int,
        processed_image_url: str,
        original_image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_settings: Optional[Dict[str, Any]] = None,
        content_sources: Optional[List[Dict[str, Any]]] = None
    ) -> BackgroundRemoval:
        """Create a new background removal record."""
        async with self.get_session() as session:
            bg_removal = BackgroundRemoval(
                user_id=user_id,
                original_image_url=original_image_url,
                processed_image_url=processed_image_url,
                metadata=metadata,
                image_settings=image_settings,
                content_sources=content_sources
            )
            session.add(bg_removal)
            await session.flush()
            await session.refresh(bg_removal)
            
            # Invalidate user's background removals cache
            await self._invalidate_cache_pattern(f"ads_db:bg_removals:*:user_id:{user_id}")
            
            return bg_removal

    async def get_background_removal(
        self,
        user_id: int,
        removal_id: int
    ) -> Optional[BackgroundRemoval]:
        """Get a background removal record by ID with caching."""
        cache_key = await self._get_cache_key("get_bg_removal", user_id=user_id, removal_id=removal_id)
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            return BackgroundRemoval(**cached_result)
        
        async with self.get_session() as session:
            result = await session.execute(
                select(BackgroundRemoval).filter(
                    and_(
                        BackgroundRemoval.id == removal_id,
                        BackgroundRemoval.user_id == user_id,
                        BackgroundRemoval.is_deleted == False
                    )
                )
            )
            bg_removal = result.scalar_one_or_none()
            
            if bg_removal:
                await self._set_cached_result(cache_key, bg_removal.__dict__)
            
            return bg_removal

    async def list_background_removals(
        self,
        user_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[BackgroundRemoval]:
        """List background removal records for a user with caching."""
        cache_key = await self._get_cache_key("list_bg_removals", user_id=user_id, limit=limit, offset=offset)
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            return [BackgroundRemoval(**item) for item in cached_result]
        
        async with self.get_session() as session:
            result = await session.execute(
                select(BackgroundRemoval).filter(
                    and_(
                        BackgroundRemoval.user_id == user_id,
                        BackgroundRemoval.is_deleted == False
                    )
                ).order_by(desc(BackgroundRemoval.created_at)).offset(offset).limit(limit)
            )
            bg_removals = result.scalars().all()
            
            # Cache the result
            cache_data = [bg.__dict__ for bg in bg_removals]
            await self._set_cached_result(cache_key, cache_data, ttl=1800)
            
            return bg_removals

    async def create_ads_analytics(
        self,
        user_id: int,
        ads_generation_id: int,
        metrics: Dict[str, Any],
        email_metrics: Optional[Dict[str, Any]] = None,
        email_settings: Optional[Dict[str, Any]] = None
    ) -> AdsAnalytics:
        """Create a new ads analytics record."""
        async with self.get_session() as session:
            analytics = AdsAnalytics(
                user_id=user_id,
                ads_generation_id=ads_generation_id,
                metrics=metrics,
                email_metrics=email_metrics,
                email_settings=email_settings
            )
            session.add(analytics)
            await session.flush()
            await session.refresh(analytics)
            
            # Invalidate analytics cache
            await self._invalidate_cache_pattern(f"ads_db:analytics:*:user_id:{user_id}")
            
            return analytics

    async def get_ads_analytics(
        self,
        user_id: int,
        analytics_id: int
    ) -> Optional[AdsAnalytics]:
        """Get an ads analytics record by ID with caching."""
        cache_key = await self._get_cache_key("get_analytics", user_id=user_id, analytics_id=analytics_id)
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            return AdsAnalytics(**cached_result)
        
        async with self.get_session() as session:
            result = await session.execute(
                select(AdsAnalytics).filter(
                    and_(
                        AdsAnalytics.id == analytics_id,
                        AdsAnalytics.user_id == user_id
                    )
                )
            )
            analytics = result.scalar_one_or_none()
            
            if analytics:
                await self._set_cached_result(cache_key, analytics.__dict__)
            
            return analytics

    async def list_ads_analytics(
        self,
        user_id: int,
        ads_generation_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AdsAnalytics]:
        """List ads analytics records for a user with caching."""
        cache_key = await self._get_cache_key(
            "list_analytics", 
            user_id=user_id, 
            ads_generation_id=ads_generation_id,
            limit=limit, 
            offset=offset
        )
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            return [AdsAnalytics(**item) for item in cached_result]
        
        async with self.get_session() as session:
            query = select(AdsAnalytics).filter(AdsAnalytics.user_id == user_id)
            
            if ads_generation_id:
                query = query.filter(AdsAnalytics.ads_generation_id == ads_generation_id)
            
            query = query.order_by(desc(AdsAnalytics.created_at)).offset(offset).limit(limit)
            result = await session.execute(query)
            analytics_list = result.scalars().all()
            
            # Cache the result
            cache_data = [analytics.__dict__ for analytics in analytics_list]
            await self._set_cached_result(cache_key, cache_data, ttl=1800)
            
            return analytics_list

    async def soft_delete_ads_generation(
        self,
        user_id: int,
        ads_id: int
    ) -> bool:
        """Soft delete an ads generation record with cache invalidation."""
        async with self.get_session() as session:
            result = await session.execute(
                update(AdsGeneration)
                .where(
                    and_(
                        AdsGeneration.id == ads_id,
                        AdsGeneration.user_id == user_id,
                        AdsGeneration.is_deleted == False
                    )
                )
                .values(is_deleted=True, updated_at=datetime.utcnow())
            )
            
            if result.rowcount > 0:
                # Invalidate caches
                await self._invalidate_cache_pattern(f"ads_db:*:user_id:{user_id}")
                await self._invalidate_cache_pattern(f"ads_db:get_ads:*:ads_id:{ads_id}")
                return True
            
            return False

    async def soft_delete_background_removal(
        self,
        user_id: int,
        removal_id: int
    ) -> bool:
        """Soft delete a background removal record with cache invalidation."""
        async with self.get_session() as session:
            result = await session.execute(
                update(BackgroundRemoval)
                .where(
                    and_(
                        BackgroundRemoval.id == removal_id,
                        BackgroundRemoval.user_id == user_id,
                        BackgroundRemoval.is_deleted == False
                    )
                )
                .values(is_deleted=True, updated_at=datetime.utcnow())
            )
            
            if result.rowcount > 0:
                # Invalidate caches
                await self._invalidate_cache_pattern(f"ads_db:bg_removals:*:user_id:{user_id}")
                await self._invalidate_cache_pattern(f"ads_db:get_bg_removal:*:removal_id:{removal_id}")
                return True
            
            return False

    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics with caching."""
        cache_key = await self._get_cache_key("user_stats", user_id=user_id)
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        async with self.get_session() as session:
            # Count ads generations
            ads_count = await session.execute(
                select(func.count(AdsGeneration.id)).filter(
                    and_(
                        AdsGeneration.user_id == user_id,
                        AdsGeneration.is_deleted == False
                    )
                )
            )
            
            # Count background removals
            bg_count = await session.execute(
                select(func.count(BackgroundRemoval.id)).filter(
                    and_(
                        BackgroundRemoval.user_id == user_id,
                        BackgroundRemoval.is_deleted == False
                    )
                )
            )
            
            # Count analytics
            analytics_count = await session.execute(
                select(func.count(AdsAnalytics.id)).filter(
                    AdsAnalytics.user_id == user_id
                )
            )
            
            stats = {
                "ads_generations_count": ads_count.scalar() or 0,
                "background_removals_count": bg_count.scalar() or 0,
                "analytics_count": analytics_count.scalar() or 0,
                "last_activity": datetime.utcnow().isoformat()
            }
            
            # Cache for 1 hour
            await self._set_cached_result(cache_key, stats, ttl=3600)
            
            return stats

    async def cleanup_old_records(self, days: int = 90) -> Dict[str, int]:
        """Clean up old records for performance optimization."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self.get_session() as session:
            # Clean up old analytics
            analytics_result = await session.execute(
                delete(AdsAnalytics).where(AdsAnalytics.created_at < cutoff_date)
            )
            
            # Clean up old soft-deleted records
            ads_result = await session.execute(
                delete(AdsGeneration).where(
                    and_(
                        AdsGeneration.is_deleted == True,
                        AdsGeneration.updated_at < cutoff_date
                    )
                )
            )
            
            bg_result = await session.execute(
                delete(BackgroundRemoval).where(
                    and_(
                        BackgroundRemoval.is_deleted == True,
                        BackgroundRemoval.updated_at < cutoff_date
                    )
                )
            )
            
            cleanup_stats = {
                "analytics_deleted": analytics_result.rowcount,
                "ads_deleted": ads_result.rowcount,
                "background_removals_deleted": bg_result.rowcount
            }
            
            # Invalidate all caches after cleanup
            await self._invalidate_cache_pattern("ads_db:*")
            
            return cleanup_stats

    async def close(self) -> Any:
        """Close database connections and cleanup resources."""
        if self._engine:
            await self._engine.dispose()
        if self._redis_client:
            await self._redis_client.close() 