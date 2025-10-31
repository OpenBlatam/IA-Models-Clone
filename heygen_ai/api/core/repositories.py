from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
from typing import Optional, List, Dict, Any, Union, TypeVar, Generic, Type
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import BaseModel
from .async_database import AsyncDatabaseManager, DatabaseConnectionPool
        from .database import User  # Import here to avoid circular imports
        from .database import Video  # Import here to avoid circular imports
        from .database import ModelUsage  # Import here to avoid circular imports
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Enhanced Async Database Repositories for HeyGen AI API
Optimized CRUD operations, bulk operations, and advanced querying
"""



logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T]):
    """
    Base repository with common CRUD operations.
    Generic type T should be a SQLAlchemy model.
    """
    
    def __init__(self, db_manager: AsyncDatabaseManager, model: Type[T]):
        
    """__init__ function."""
self.db_manager = db_manager
        self.model = model
    
    async def create(self, **kwargs) -> T:
        """Create a new record."""
        try:
            async with self.db_manager.get_session() as session:
                instance = self.model(**kwargs)
                session.add(instance)
                await session.commit()
                await session.refresh(instance)
                return instance
        except IntegrityError as e:
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise
    
    async def get_by_id(self, id: int) -> Optional[T]:
        """Get record by ID."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(self.model).where(self.model.id == id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} by ID {id}: {e}")
            raise
    
    async def get_by_field(self, field: str, value: Any) -> Optional[T]:
        """Get record by field value."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(self.model).where(getattr(self.model, field) == value)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} by {field}={value}: {e}")
            raise
    
    async def get_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        order_desc: bool = True
    ) -> List[T]:
        """Get all records with optional pagination and ordering."""
        try:
            async with self.db_manager.get_session() as session:
                query = select(self.model)
                
                if order_by:
                    order_column = getattr(self.model, order_by)
                    if order_desc:
                        query = query.order_by(desc(order_column))
                    else:
                        query = query.order_by(asc(order_column))
                
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise
    
    async def update(self, id: int, **kwargs) -> Optional[T]:
        """Update record by ID."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(self.model).where(self.model.id == id)
                )
                instance = result.scalar_one_or_none()
                
                if not instance:
                    return None
                
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                
                await session.commit()
                await session.refresh(instance)
                return instance
        except Exception as e:
            logger.error(f"Error updating {self.model.__name__} {id}: {e}")
            raise
    
    async def delete(self, id: int) -> bool:
        """Delete record by ID."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(self.model).where(self.model.id == id)
                )
                instance = result.scalar_one_or_none()
                
                if not instance:
                    return False
                
                await session.delete(instance)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting {self.model.__name__} {id}: {e}")
            raise
    
    async def count(self, **filters) -> int:
        """Count records with optional filters."""
        try:
            async with self.db_manager.get_session() as session:
                query = select(func.count(self.model.id))
                
                if filters:
                    conditions = []
                    for field, value in filters.items():
                        if hasattr(self.model, field):
                            conditions.append(getattr(self.model, field) == value)
                    if conditions:
                        query = query.where(and_(*conditions))
                
                result = await session.execute(query)
                return result.scalar()
        except Exception as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            raise
    
    async def exists(self, **filters) -> bool:
        """Check if record exists with given filters."""
        return await self.count(**filters) > 0
    
    async def bulk_create(self, data_list: List[Dict[str, Any]]) -> List[T]:
        """Create multiple records in bulk."""
        try:
            async with self.db_manager.get_session() as session:
                instances = [self.model(**data) for data in data_list]
                session.add_all(instances)
                await session.commit()
                
                # Refresh all instances to get their IDs
                for instance in instances:
                    await session.refresh(instance)
                
                return instances
        except Exception as e:
            logger.error(f"Error bulk creating {self.model.__name__}: {e}")
            raise
    
    async def bulk_update(self, id_list: List[int], **kwargs) -> int:
        """Update multiple records by IDs."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    update(self.model)
                    .where(self.model.id.in_(id_list))
                    .values(**kwargs)
                )
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error bulk updating {self.model.__name__}: {e}")
            raise
    
    async def bulk_delete(self, id_list: List[int]) -> int:
        """Delete multiple records by IDs."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    delete(self.model).where(self.model.id.in_(id_list))
                )
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error bulk deleting {self.model.__name__}: {e}")
            raise


class UserRepository(BaseRepository):
    """Enhanced user repository with user-specific operations."""
    
    async async def get_by_api_key(self, api_key: str) -> Optional[Any]:
        """Get user by API key."""
        return await self.get_by_field("api_key", api_key)
    
    async def get_by_email(self, email: str) -> Optional[Any]:
        """Get user by email."""
        return await self.get_by_field("email", email)
    
    async def get_by_username(self, username: str) -> Optional[Any]:
        """Get user by username."""
        return await self.get_by_field("username", username)
    
    async def get_active_users(self, limit: Optional[int] = None) -> List[Any]:
        """Get all active users."""
        try:
            async with self.db_manager.get_session() as session:
                query = select(self.model).where(self.model.is_active == True)
                if limit:
                    query = query.limit(limit)
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            raise
    
    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp."""
        return await self.update(user_id, updated_at=datetime.utcnow()) is not None
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics."""
        try:
            async with self.db_manager.get_session() as session:
                # Get user with related videos
                result = await session.execute(
                    select(self.model)
                    .options(selectinload(self.model.videos))
                    .where(self.model.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    return {}
                
                videos = user.videos
                total_videos = len(videos)
                completed_videos = len([v for v in videos if v.status == "completed"])
                failed_videos = len([v for v in videos if v.status == "failed"])
                
                return {
                    "user_id": user_id,
                    "total_videos": total_videos,
                    "completed_videos": completed_videos,
                    "failed_videos": failed_videos,
                    "success_rate": (completed_videos / total_videos * 100) if total_videos > 0 else 0,
                    "created_at": user.created_at,
                    "last_updated": user.updated_at
                }
        except Exception as e:
            logger.error(f"Error getting user stats for {user_id}: {e}")
            raise


class VideoRepository(BaseRepository):
    """Enhanced video repository with video-specific operations."""
    
    async def get_by_video_id(self, video_id: str) -> Optional[Any]:
        """Get video by video_id field."""
        return await self.get_by_field("video_id", video_id)
    
    async def get_user_videos(
        self, 
        user_id: int, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Any]:
        """Get videos for a specific user with optional filtering."""
        try:
            async with self.db_manager.get_session() as session:
                query = select(self.model).where(self.model.user_id == user_id)
                
                if status:
                    query = query.where(self.model.status == status)
                
                query = query.order_by(desc(self.model.created_at))
                
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting videos for user {user_id}: {e}")
            raise
    
    async def update_status(self, video_id: str, status: str, **kwargs) -> bool:
        """Update video status and other fields."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(self.model).where(self.model.video_id == video_id)
                )
                video = result.scalar_one_or_none()
                
                if not video:
                    return False
                
                video.status = status
                for key, value in kwargs.items():
                    if hasattr(video, key):
                        setattr(video, key, value)
                
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating video status {video_id}: {e}")
            raise
    
    async def get_videos_by_status(self, status: str, limit: Optional[int] = None) -> List[Any]:
        """Get videos by status."""
        try:
            async with self.db_manager.get_session() as session:
                query = select(self.model).where(self.model.status == status)
                if limit:
                    query = query.limit(limit)
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting videos by status {status}: {e}")
            raise
    
    async def get_processing_videos(self) -> List[Any]:
        """Get all videos currently being processed."""
        return await self.get_videos_by_status("processing")
    
    async def get_completed_videos(self, limit: Optional[int] = None) -> List[Any]:
        """Get recently completed videos."""
        return await self.get_videos_by_status("completed", limit)
    
    async def get_failed_videos(self, limit: Optional[int] = None) -> List[Any]:
        """Get recently failed videos."""
        return await self.get_videos_by_status("failed", limit)
    
    async def get_video_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get video statistics."""
        try:
            async with self.db_manager.get_session() as session:
                # Base query
                query = select(self.model)
                if user_id:
                    query = query.where(self.model.user_id == user_id)
                
                # Get total count
                total_result = await session.execute(
                    select(func.count(self.model.id)).select_from(query.subquery())
                )
                total_videos = total_result.scalar()
                
                # Get status counts
                status_counts = {}
                for status in ["processing", "completed", "failed"]:
                    status_query = select(func.count(self.model.id)).where(self.model.status == status)
                    if user_id:
                        status_query = status_query.where(self.model.user_id == user_id)
                    result = await session.execute(status_query)
                    status_counts[status] = result.scalar()
                
                # Get average processing time
                avg_time_query = select(func.avg(self.model.processing_time)).where(
                    and_(self.model.status == "completed", self.model.processing_time.isnot(None))
                )
                if user_id:
                    avg_time_query = avg_time_query.where(self.model.user_id == user_id)
                avg_result = await session.execute(avg_time_query)
                avg_processing_time = avg_result.scalar() or 0.0
                
                return {
                    "total_videos": total_videos,
                    "status_counts": status_counts,
                    "average_processing_time": avg_processing_time,
                    "success_rate": (status_counts["completed"] / total_videos * 100) if total_videos > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error getting video stats: {e}")
            raise
    
    async def cleanup_old_videos(self, days: int = 30) -> int:
        """Delete videos older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    delete(self.model).where(self.model.created_at < cutoff_date)
                )
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up old videos: {e}")
            raise


class ModelUsageRepository(BaseRepository):
    """Enhanced model usage repository for analytics."""
    
    async def log_usage(self, usage_data: Dict[str, Any]) -> Any:
        """Log model usage."""
        return await self.create(**usage_data)
    
    async def get_usage_stats(
        self, 
        user_id: Optional[int] = None, 
        days: int = 30,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage statistics."""
        try:
            async with self.db_manager.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Base query
                query = select(self.model).where(self.model.created_at >= cutoff_date)
                if user_id:
                    query = query.where(self.model.user_id == user_id)
                if model_type:
                    query = query.where(self.model.model_type == model_type)
                
                # Get total usage
                total_result = await session.execute(
                    select(func.count(self.model.id)).select_from(query.subquery())
                )
                total_usage = total_result.scalar()
                
                # Get average processing time
                avg_time_query = select(func.avg(self.model.processing_time)).select_from(query.subquery())
                avg_result = await session.execute(avg_time_query)
                avg_processing_time = avg_result.scalar() or 0.0
                
                # Get average memory usage
                avg_memory_query = select(func.avg(self.model.memory_usage)).select_from(query.subquery())
                avg_memory_result = await session.execute(avg_memory_query)
                avg_memory_usage = avg_memory_result.scalar() or 0.0
                
                # Get average GPU usage
                avg_gpu_query = select(func.avg(self.model.gpu_usage)).select_from(query.subquery())
                avg_gpu_result = await session.execute(avg_gpu_query)
                avg_gpu_usage = avg_gpu_result.scalar() or 0.0
                
                return {
                    "total_usage": total_usage,
                    "average_processing_time": avg_processing_time,
                    "average_memory_usage": avg_memory_usage,
                    "average_gpu_usage": avg_gpu_usage,
                    "period_days": days
                }
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            raise
    
    async def get_daily_usage(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily usage statistics."""
        try:
            async with self.db_manager.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Group by date and count
                result = await session.execute(
                    select(
                        func.date(self.model.created_at).label('date'),
                        func.count(self.model.id).label('count'),
                        func.avg(self.model.processing_time).label('avg_time')
                    )
                    .where(self.model.created_at >= cutoff_date)
                    .group_by(func.date(self.model.created_at))
                    .order_by(desc(func.date(self.model.created_at)))
                )
                
                return [
                    {
                        "date": row.date,
                        "count": row.count,
                        "average_processing_time": row.avg_time or 0.0
                    }
                    for row in result
                ]
        except Exception as e:
            logger.error(f"Error getting daily usage: {e}")
            raise


# Repository factory
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    def __init__(self, db_pool: DatabaseConnectionPool):
        
    """__init__ function."""
self.db_pool = db_pool
    
    def get_user_repository(self, db_name: Optional[str] = None) -> UserRepository:
        """Get user repository instance."""
        # This would need to be adapted based on how you want to handle multiple databases
        # For now, using the primary database
        if not self.db_pool.primary_db:
            raise RuntimeError("No primary database configured")
        
        db_manager = self.db_pool.databases[self.db_pool.primary_db]
        return UserRepository(db_manager, User)
    
    def get_video_repository(self, db_name: Optional[str] = None) -> VideoRepository:
        """Get video repository instance."""
        if not self.db_pool.primary_db:
            raise RuntimeError("No primary database configured")
        
        db_manager = self.db_pool.databases[self.db_pool.primary_db]
        return VideoRepository(db_manager, Video)
    
    def get_model_usage_repository(self, db_name: Optional[str] = None) -> ModelUsageRepository:
        """Get model usage repository instance."""
        if not self.db_pool.primary_db:
            raise RuntimeError("No primary database configured")
        
        db_manager = self.db_pool.databases[self.db_pool.primary_db]
        return ModelUsageRepository(db_manager, ModelUsage) 