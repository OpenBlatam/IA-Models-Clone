from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
from typing import Optional, List, Dict, Any, Union, TypeVar, Generic, Type, Sequence
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload, joinedload, load_only, contains_eager
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, NoResultFound, MultipleResultsFound
from sqlalchemy.sql import Select
from pydantic import BaseModel
from contextlib import asynccontextmanager
from .sqlalchemy_config import SQLAlchemyManager
from .models.sqlalchemy_models import Base, User, Video, ModelUsage
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Enhanced SQLAlchemy 2.0 Repositories for HeyGen AI API
Modern repository pattern with SQLAlchemy 2.0 features, type annotations, and performance optimization.
"""



logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """
    Enhanced base repository with SQLAlchemy 2.0 features.
    Generic type T should be a SQLAlchemy model.
    """
    
    def __init__(self, sqlalchemy_manager: SQLAlchemyManager, model: Type[T]):
        
    """__init__ function."""
self.sqlalchemy_manager = sqlalchemy_manager
        self.model = model
    
    @asynccontextmanager
    async def _get_session(self) -> AsyncSession:
        """Get database session with automatic cleanup."""
        async with self.sqlalchemy_manager.get_session() as session:
            yield session
    
    async def create(self, **kwargs) -> T:
        """Create a new record with enhanced error handling."""
        try:
            async with self._get_session() as session:
                instance = self.model(**kwargs)
                session.add(instance)
                await session.commit()
                await session.refresh(instance)
                return instance
        except IntegrityError as e:
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise ValueError(f"Data validation failed: {e}")
        except Exception as e:
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise
    
    async def get_by_id(self, id: int) -> Optional[T]:
        """Get record by ID with optional relationship loading."""
        try:
            async with self._get_session() as session:
                stmt = select(self.model).where(self.model.id == id)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} by ID {id}: {e}")
            raise
    
    async def get_by_field(self, field: str, value: Any) -> Optional[T]:
        """Get record by field value."""
        try:
            async with self._get_session() as session:
                field_attr = getattr(self.model, field)
                stmt = select(self.model).where(field_attr == value)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} by {field}={value}: {e}")
            raise
    
    async def get_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        load_relationships: Optional[List[str]] = None
    ) -> List[T]:
        """Get all records with enhanced options."""
        try:
            async with self._get_session() as session:
                stmt = select(self.model)
                
                # Add relationship loading
                if load_relationships:
                    for rel in load_relationships:
                        if hasattr(self.model, rel):
                            stmt = stmt.options(selectinload(getattr(self.model, rel)))
                
                # Add ordering
                if order_by:
                    order_column = getattr(self.model, order_by)
                    if order_desc:
                        stmt = stmt.order_by(desc(order_column))
                    else:
                        stmt = stmt.order_by(asc(order_column))
                
                # Add pagination
                if limit:
                    stmt = stmt.limit(limit)
                if offset:
                    stmt = stmt.offset(offset)
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise
    
    async def update(self, id: int, **kwargs) -> Optional[T]:
        """Update record by ID with validation."""
        try:
            async with self._get_session() as session:
                stmt = select(self.model).where(self.model.id == id)
                result = await session.execute(stmt)
                instance = result.scalar_one_or_none()
                
                if not instance:
                    return None
                
                # Update fields with validation
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
            async with self._get_session() as session:
                stmt = select(self.model).where(self.model.id == id)
                result = await session.execute(stmt)
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
            async with self._get_session() as session:
                stmt = select(func.count(self.model.id))
                
                if filters:
                    conditions = []
                    for field, value in filters.items():
                        if hasattr(self.model, field):
                            conditions.append(getattr(self.model, field) == value)
                    if conditions:
                        stmt = stmt.where(and_(*conditions))
                
                result = await session.execute(stmt)
                return result.scalar()
        except Exception as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            raise
    
    async def exists(self, **filters) -> bool:
        """Check if record exists with given filters."""
        return await self.count(**filters) > 0
    
    async def bulk_create(self, data_list: List[Dict[str, Any]]) -> List[T]:
        """Create multiple records in bulk with enhanced performance."""
        try:
            async with self._get_session() as session:
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
            async with self._get_session() as session:
                stmt = (
                    update(self.model)
                    .where(self.model.id.in_(id_list))
                    .values(**kwargs)
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error bulk updating {self.model.__name__}: {e}")
            raise
    
    async def bulk_delete(self, id_list: List[int]) -> int:
        """Delete multiple records by IDs."""
        try:
            async with self._get_session() as session:
                stmt = delete(self.model).where(self.model.id.in_(id_list))
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error bulk deleting {self.model.__name__}: {e}")
            raise
    
    async def find_by_criteria(
        self,
        criteria: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        order_desc: bool = True
    ) -> List[T]:
        """Find records by multiple criteria."""
        try:
            async with self._get_session() as session:
                stmt = select(self.model)
                
                # Build conditions
                conditions = []
                for field, value in criteria.items():
                    if hasattr(self.model, field):
                        field_attr = getattr(self.model, field)
                        if isinstance(value, (list, tuple)):
                            conditions.append(field_attr.in_(value))
                        elif isinstance(value, dict):
                            # Handle range queries
                            if 'min' in value:
                                conditions.append(field_attr >= value['min'])
                            if 'max' in value:
                                conditions.append(field_attr <= value['max'])
                        else:
                            conditions.append(field_attr == value)
                
                if conditions:
                    stmt = stmt.where(and_(*conditions))
                
                # Add ordering
                if order_by:
                    order_column = getattr(self.model, order_by)
                    if order_desc:
                        stmt = stmt.order_by(desc(order_column))
                    else:
                        stmt = stmt.order_by(asc(order_column))
                
                # Add pagination
                if limit:
                    stmt = stmt.limit(limit)
                if offset:
                    stmt = stmt.offset(offset)
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error finding {self.model.__name__} by criteria: {e}")
            raise


class UserRepository(BaseRepository[User]):
    """Enhanced user repository with user-specific operations."""
    
    async async def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key with validation."""
        return await self.get_by_field("api_key", api_key)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return await self.get_by_field("email", email.lower())
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return await self.get_by_field("username", username.lower())
    
    async def get_active_users(self, limit: Optional[int] = None) -> List[User]:
        """Get all active users."""
        try:
            async with self._get_session() as session:
                stmt = (
                    select(User)
                    .where(User.is_active == True, User.is_deleted == False)
                    .order_by(desc(User.created_at))
                )
                
                if limit:
                    stmt = stmt.limit(limit)
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            raise
    
    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp."""
        try:
            async with self._get_session() as session:
                stmt = (
                    update(User)
                    .where(User.id == user_id)
                    .values(
                        last_login_at=datetime.now(),
                        login_count=User.login_count + 1
                    )
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating last login for user {user_id}: {e}")
            raise
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        try:
            async with self._get_session() as session:
                # Get user with related data
                stmt = (
                    select(User)
                    .options(selectinload(User.videos), selectinload(User.model_usage))
                    .where(User.id == user_id)
                )
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()
                
                if not user:
                    return {}
                
                # Calculate statistics
                videos = user.videos
                total_videos = len(videos)
                completed_videos = len([v for v in videos if v.status == "completed"])
                failed_videos = len([v for v in videos if v.status == "failed"])
                processing_videos = len([v for v in videos if v.status == "processing"])
                
                # Calculate processing metrics
                total_processing_time = sum(v.processing_time or 0 for v in videos if v.processing_time)
                avg_processing_time = total_processing_time / max(completed_videos, 1)
                
                # Calculate usage metrics
                model_usage = user.model_usage
                total_model_usage = len(model_usage)
                successful_usage = len([u for u in model_usage if u.success])
                
                return {
                    "user_id": user_id,
                    "username": user.username,
                    "email": user.email,
                    "is_active": user.is_active,
                    "created_at": user.created_at,
                    "last_login_at": user.last_login_at,
                    "login_count": user.login_count,
                    "videos": {
                        "total": total_videos,
                        "completed": completed_videos,
                        "failed": failed_videos,
                        "processing": processing_videos,
                        "success_rate": (completed_videos / total_videos * 100) if total_videos > 0 else 0,
                        "average_processing_time": avg_processing_time
                    },
                    "model_usage": {
                        "total": total_model_usage,
                        "successful": successful_usage,
                        "success_rate": (successful_usage / total_model_usage * 100) if total_model_usage > 0 else 0
                    }
                }
        except Exception as e:
            logger.error(f"Error getting user stats for {user_id}: {e}")
            raise
    
    async def search_users(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[User]:
        """Search users by username or email."""
        try:
            async with self._get_session() as session:
                search_term = f"%{query.lower()}%"
                stmt = (
                    select(User)
                    .where(
                        and_(
                            User.is_deleted == False,
                            or_(
                                func.lower(User.username).like(search_term),
                                func.lower(User.email).like(search_term)
                            )
                        )
                    )
                    .order_by(desc(User.created_at))
                    .limit(limit)
                    .offset(offset)
                )
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error searching users: {e}")
            raise


class VideoRepository(BaseRepository[Video]):
    """Enhanced video repository with video-specific operations."""
    
    async def get_by_video_id(self, video_id: str) -> Optional[Video]:
        """Get video by video_id field."""
        return await self.get_by_field("video_id", video_id)
    
    async def get_user_videos(
        self, 
        user_id: int, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        status: Optional[str] = None,
        include_relationships: bool = True
    ) -> List[Video]:
        """Get videos for a specific user with enhanced options."""
        try:
            async with self._get_session() as session:
                stmt = select(Video).where(Video.user_id == user_id)
                
                if status:
                    stmt = stmt.where(Video.status == status)
                
                if include_relationships:
                    stmt = stmt.options(selectinload(Video.user))
                
                stmt = stmt.order_by(desc(Video.created_at))
                
                if limit:
                    stmt = stmt.limit(limit)
                if offset:
                    stmt = stmt.offset(offset)
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting videos for user {user_id}: {e}")
            raise
    
    async def update_status(self, video_id: str, status: str, **kwargs) -> bool:
        """Update video status and other fields."""
        try:
            async with self._get_session() as session:
                stmt = (
                    update(Video)
                    .where(Video.video_id == video_id)
                    .values(status=status, **kwargs)
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating video status {video_id}: {e}")
            raise
    
    async def get_videos_by_status(self, status: str, limit: Optional[int] = None) -> List[Video]:
        """Get videos by status."""
        try:
            async with self._get_session() as session:
                stmt = (
                    select(Video)
                    .where(Video.status == status)
                    .order_by(desc(Video.created_at))
                )
                
                if limit:
                    stmt = stmt.limit(limit)
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting videos by status {status}: {e}")
            raise
    
    async def get_processing_videos(self) -> List[Video]:
        """Get all videos currently being processed."""
        return await self.get_videos_by_status("processing")
    
    async def get_completed_videos(self, limit: Optional[int] = None) -> List[Video]:
        """Get recently completed videos."""
        return await self.get_videos_by_status("completed", limit)
    
    async def get_failed_videos(self, limit: Optional[int] = None) -> List[Video]:
        """Get recently failed videos."""
        return await self.get_videos_by_status("failed", limit)
    
    async def get_video_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive video statistics."""
        try:
            async with self._get_session() as session:
                # Base query
                stmt = select(Video)
                if user_id:
                    stmt = stmt.where(Video.user_id == user_id)
                
                # Get total count
                total_result = await session.execute(
                    select(func.count(Video.id)).select_from(stmt.subquery())
                )
                total_videos = total_result.scalar()
                
                # Get status counts
                status_counts = {}
                for status in ["pending", "processing", "completed", "failed", "cancelled"]:
                    status_stmt = select(func.count(Video.id)).where(Video.status == status)
                    if user_id:
                        status_stmt = status_stmt.where(Video.user_id == user_id)
                    result = await session.execute(status_stmt)
                    status_counts[status] = result.scalar()
                
                # Get average processing time
                avg_time_stmt = (
                    select(func.avg(Video.processing_time))
                    .where(
                        and_(
                            Video.status == "completed",
                            Video.processing_time.isnot(None)
                        )
                    )
                )
                if user_id:
                    avg_time_stmt = avg_time_stmt.where(Video.user_id == user_id)
                avg_result = await session.execute(avg_time_stmt)
                avg_processing_time = avg_result.scalar() or 0.0
                
                # Get quality distribution
                quality_stmt = (
                    select(Video.quality, func.count(Video.id))
                    .group_by(Video.quality)
                )
                if user_id:
                    quality_stmt = quality_stmt.where(Video.user_id == user_id)
                quality_result = await session.execute(quality_stmt)
                quality_distribution = dict(quality_result.all())
                
                return {
                    "total_videos": total_videos,
                    "status_counts": status_counts,
                    "average_processing_time": avg_processing_time,
                    "success_rate": (status_counts["completed"] / total_videos * 100) if total_videos > 0 else 0,
                    "quality_distribution": quality_distribution
                }
        except Exception as e:
            logger.error(f"Error getting video stats: {e}")
            raise
    
    async def cleanup_old_videos(self, days: int = 30) -> int:
        """Delete videos older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            async with self._get_session() as session:
                stmt = delete(Video).where(Video.created_at < cutoff_date)
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up old videos: {e}")
            raise
    
    async def get_videos_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[int] = None
    ) -> List[Video]:
        """Get videos within a date range."""
        try:
            async with self._get_session() as session:
                stmt = (
                    select(Video)
                    .where(
                        and_(
                            Video.created_at >= start_date,
                            Video.created_at <= end_date
                        )
                    )
                    .order_by(desc(Video.created_at))
                )
                
                if user_id:
                    stmt = stmt.where(Video.user_id == user_id)
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting videos by date range: {e}")
            raise


class ModelUsageRepository(BaseRepository[ModelUsage]):
    """Enhanced model usage repository for analytics."""
    
    async def log_usage(self, usage_data: Dict[str, Any]) -> ModelUsage:
        """Log model usage with validation."""
        return await self.create(**usage_data)
    
    async def get_usage_stats(
        self, 
        user_id: Optional[int] = None, 
        days: int = 30,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        try:
            async with self._get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Base query
                stmt = select(ModelUsage).where(ModelUsage.created_at >= cutoff_date)
                if user_id:
                    stmt = stmt.where(ModelUsage.user_id == user_id)
                if model_type:
                    stmt = stmt.where(ModelUsage.model_type == model_type)
                
                # Get total usage
                total_result = await session.execute(
                    select(func.count(ModelUsage.id)).select_from(stmt.subquery())
                )
                total_usage = total_result.scalar()
                
                # Get average processing time
                avg_time_stmt = select(func.avg(ModelUsage.processing_time)).select_from(stmt.subquery())
                avg_result = await session.execute(avg_time_stmt)
                avg_processing_time = avg_result.scalar() or 0.0
                
                # Get average memory usage
                avg_memory_stmt = select(func.avg(ModelUsage.memory_usage)).select_from(stmt.subquery())
                avg_memory_result = await session.execute(avg_memory_stmt)
                avg_memory_usage = avg_memory_result.scalar() or 0.0
                
                # Get average GPU usage
                avg_gpu_stmt = select(func.avg(ModelUsage.gpu_usage)).select_from(stmt.subquery())
                avg_gpu_result = await session.execute(avg_gpu_stmt)
                avg_gpu_usage = avg_gpu_result.scalar() or 0.0
                
                # Get success rate
                success_stmt = (
                    select(func.count(ModelUsage.id))
                    .where(
                        and_(
                            ModelUsage.created_at >= cutoff_date,
                            ModelUsage.success == True
                        )
                    )
                )
                if user_id:
                    success_stmt = success_stmt.where(ModelUsage.user_id == user_id)
                if model_type:
                    success_stmt = success_stmt.where(ModelUsage.model_type == model_type)
                
                success_result = await session.execute(success_stmt)
                successful_usage = success_result.scalar()
                
                return {
                    "total_usage": total_usage,
                    "successful_usage": successful_usage,
                    "success_rate": (successful_usage / total_usage * 100) if total_usage > 0 else 0,
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
            async with self._get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Group by date and count
                stmt = (
                    select(
                        func.date(ModelUsage.created_at).label('date'),
                        func.count(ModelUsage.id).label('count'),
                        func.avg(ModelUsage.processing_time).label('avg_time'),
                        func.avg(ModelUsage.memory_usage).label('avg_memory'),
                        func.avg(ModelUsage.gpu_usage).label('avg_gpu')
                    )
                    .where(ModelUsage.created_at >= cutoff_date)
                    .group_by(func.date(ModelUsage.created_at))
                    .order_by(desc(func.date(ModelUsage.created_at)))
                )
                
                result = await session.execute(stmt)
                
                return [
                    {
                        "date": row.date,
                        "count": row.count,
                        "average_processing_time": row.avg_time or 0.0,
                        "average_memory_usage": row.avg_memory or 0.0,
                        "average_gpu_usage": row.avg_gpu or 0.0
                    }
                    for row in result
                ]
        except Exception as e:
            logger.error(f"Error getting daily usage: {e}")
            raise
    
    async def get_model_performance(self, model_type: str, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for a specific model type."""
        try:
            async with self._get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                stmt = (
                    select(
                        func.count(ModelUsage.id).label('total_usage'),
                        func.avg(ModelUsage.processing_time).label('avg_time'),
                        func.avg(ModelUsage.memory_usage).label('avg_memory'),
                        func.avg(ModelUsage.gpu_usage).label('avg_gpu'),
                        func.avg(ModelUsage.accuracy).label('avg_accuracy'),
                        func.avg(ModelUsage.confidence).label('avg_confidence')
                    )
                    .where(
                        and_(
                            ModelUsage.model_type == model_type,
                            ModelUsage.created_at >= cutoff_date
                        )
                    )
                )
                
                result = await session.execute(stmt)
                row = result.first()
                
                if not row:
                    return {}
                
                return {
                    "model_type": model_type,
                    "total_usage": row.total_usage,
                    "average_processing_time": row.avg_time or 0.0,
                    "average_memory_usage": row.avg_memory or 0.0,
                    "average_gpu_usage": row.avg_gpu or 0.0,
                    "average_accuracy": row.avg_accuracy or 0.0,
                    "average_confidence": row.avg_confidence or 0.0,
                    "period_days": days
                }
        except Exception as e:
            logger.error(f"Error getting model performance for {model_type}: {e}")
            raise


# Repository factory
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    def __init__(self, sqlalchemy_manager: SQLAlchemyManager):
        
    """__init__ function."""
self.sqlalchemy_manager = sqlalchemy_manager
    
    def get_user_repository(self) -> UserRepository:
        """Get user repository instance."""
        return UserRepository(self.sqlalchemy_manager, User)
    
    def get_video_repository(self) -> VideoRepository:
        """Get video repository instance."""
        return VideoRepository(self.sqlalchemy_manager, Video)
    
    def get_model_usage_repository(self) -> ModelUsageRepository:
        """Get model usage repository instance."""
        return ModelUsageRepository(self.sqlalchemy_manager, ModelUsage)
    
    def get_repository(self, model_class: Type[T]) -> BaseRepository[T]:
        """Get repository for any model class."""
        return BaseRepository(self.sqlalchemy_manager, model_class) 