from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import List, Optional, Dict, Any, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, desc, asc
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
from core.exceptions import DatabaseError, ValidationError, handle_async_exception
from core.types import ProcessingStatus, TaskPriority
from .models import User, VideoRequest, ProcessingTask, UploadedFile, SystemMetrics, CacheEntry
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Repository pattern for OS Content UGC Video Generator
Data access layer with async support
"""



logger = structlog.get_logger("os_content.repository")

class BaseRepository:
    """Base repository with common CRUD operations"""
    
    def __init__(self, session: AsyncSession):
        
    """__init__ function."""
self.session = session
    
    async def commit(self) -> Any:
        """Commit current transaction"""
        await self.session.commit()
    
    async def rollback(self) -> Any:
        """Rollback current transaction"""
        await self.session.rollback()

class UserRepository(BaseRepository):
    """User repository for user management"""
    
    async def create_user(self, username: str, email: str) -> User:
        """Create a new user"""
        try:
            user = User(username=username, email=email)
            self.session.add(user)
            await self.commit()
            return user
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to create user: {e}")
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise DatabaseError(f"Failed to get user: {e}")
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise DatabaseError(f"Failed to get user by email: {e}")
    
    async def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user"""
        try:
            result = await self.session.execute(
                update(User)
                .where(User.id == user_id)
                .values(**kwargs, updated_at=datetime.utcnow())
            )
            await self.commit()
            
            if result.rowcount > 0:
                return await self.get_user_by_id(user_id)
            return None
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to update user: {e}")
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        try:
            result = await self.session.execute(
                delete(User).where(User.id == user_id)
            )
            await self.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to delete user: {e}")

class VideoRepository(BaseRepository):
    """Video repository for video request management"""
    
    async async def create_video_request(self, **kwargs) -> VideoRequest:
        """Create a new video request"""
        try:
            video_request = VideoRequest(**kwargs)
            self.session.add(video_request)
            await self.commit()
            return video_request
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to create video request: {e}")
    
    async async def get_video_request_by_id(self, request_id: str) -> Optional[VideoRequest]:
        """Get video request by ID with relationships"""
        try:
            result = await self.session.execute(
                select(VideoRequest)
                .options(
                    selectinload(VideoRequest.user),
                    selectinload(VideoRequest.processing_tasks),
                    selectinload(VideoRequest.uploaded_files)
                )
                .where(VideoRequest.id == request_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise DatabaseError(f"Failed to get video request: {e}")
    
    async async def get_video_requests_by_user(self, user_id: str, limit: int = 50, offset: int = 0) -> List[VideoRequest]:
        """Get video requests by user"""
        try:
            result = await self.session.execute(
                select(VideoRequest)
                .where(VideoRequest.user_id == user_id)
                .order_by(desc(VideoRequest.created_at))
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get video requests by user: {e}")
    
    async async def update_video_request(self, request_id: str, **kwargs) -> Optional[VideoRequest]:
        """Update video request"""
        try:
            result = await self.session.execute(
                update(VideoRequest)
                .where(VideoRequest.id == request_id)
                .values(**kwargs, updated_at=datetime.utcnow())
            )
            await self.commit()
            
            if result.rowcount > 0:
                return await self.get_video_request_by_id(request_id)
            return None
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to update video request: {e}")
    
    async async def get_video_requests_by_status(self, status: str, limit: int = 100) -> List[VideoRequest]:
        """Get video requests by status"""
        try:
            result = await self.session.execute(
                select(VideoRequest)
                .where(VideoRequest.status == status)
                .order_by(asc(VideoRequest.created_at))
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get video requests by status: {e}")
    
    async async def get_processing_video_requests(self) -> List[VideoRequest]:
        """Get video requests that are currently processing"""
        try:
            result = await self.session.execute(
                select(VideoRequest)
                .where(VideoRequest.status.in_([ProcessingStatus.QUEUED, ProcessingStatus.PROCESSING]))
                .order_by(asc(VideoRequest.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get processing video requests: {e}")
    
    async async def delete_video_request(self, request_id: str) -> bool:
        """Delete video request"""
        try:
            result = await self.session.execute(
                delete(VideoRequest).where(VideoRequest.id == request_id)
            )
            await self.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to delete video request: {e}")

class TaskRepository(BaseRepository):
    """Task repository for processing task management"""
    
    async def create_task(self, **kwargs) -> ProcessingTask:
        """Create a new processing task"""
        try:
            task = ProcessingTask(**kwargs)
            self.session.add(task)
            await self.commit()
            return task
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to create task: {e}")
    
    async def get_task_by_id(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task by ID"""
        try:
            result = await self.session.execute(
                select(ProcessingTask)
                .options(selectinload(ProcessingTask.video_request))
                .where(ProcessingTask.id == task_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise DatabaseError(f"Failed to get task: {e}")
    
    async async def get_tasks_by_video_request(self, video_request_id: str) -> List[ProcessingTask]:
        """Get tasks by video request ID"""
        try:
            result = await self.session.execute(
                select(ProcessingTask)
                .where(ProcessingTask.video_request_id == video_request_id)
                .order_by(desc(ProcessingTask.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get tasks by video request: {e}")
    
    async def get_pending_tasks(self, limit: int = 50) -> List[ProcessingTask]:
        """Get pending tasks ordered by priority"""
        try:
            result = await self.session.execute(
                select(ProcessingTask)
                .where(ProcessingTask.status == "pending")
                .order_by(
                    asc(ProcessingTask.priority),
                    asc(ProcessingTask.created_at)
                )
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get pending tasks: {e}")
    
    async def update_task(self, task_id: str, **kwargs) -> Optional[ProcessingTask]:
        """Update task"""
        try:
            result = await self.session.execute(
                update(ProcessingTask)
                .where(ProcessingTask.id == task_id)
                .values(**kwargs, updated_at=datetime.utcnow())
            )
            await self.commit()
            
            if result.rowcount > 0:
                return await self.get_task_by_id(task_id)
            return None
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to update task: {e}")
    
    async def mark_task_completed(self, task_id: str, result_data: Dict[str, Any] = None) -> bool:
        """Mark task as completed"""
        try:
            result = await self.session.execute(
                update(ProcessingTask)
                .where(ProcessingTask.id == task_id)
                .values(
                    status="completed",
                    completed_at=datetime.utcnow(),
                    result_data=result_data,
                    updated_at=datetime.utcnow()
                )
            )
            await self.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to mark task completed: {e}")
    
    async def mark_task_failed(self, task_id: str, error_message: str) -> bool:
        """Mark task as failed"""
        try:
            result = await self.session.execute(
                update(ProcessingTask)
                .where(ProcessingTask.id == task_id)
                .values(
                    status="failed",
                    completed_at=datetime.utcnow(),
                    error_message=error_message,
                    updated_at=datetime.utcnow()
                )
            )
            await self.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to mark task failed: {e}")

class FileRepository(BaseRepository):
    """File repository for uploaded file management"""
    
    async async def create_uploaded_file(self, **kwargs) -> UploadedFile:
        """Create a new uploaded file record"""
        try:
            uploaded_file = UploadedFile(**kwargs)
            self.session.add(uploaded_file)
            await self.commit()
            return uploaded_file
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to create uploaded file: {e}")
    
    async async def get_files_by_video_request(self, video_request_id: str) -> List[UploadedFile]:
        """Get uploaded files by video request ID"""
        try:
            result = await self.session.execute(
                select(UploadedFile)
                .where(UploadedFile.video_request_id == video_request_id)
                .order_by(UploadedFile.uploaded_at)
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get uploaded files: {e}")
    
    async def get_files_by_type(self, video_request_id: str, file_type: str) -> List[UploadedFile]:
        """Get uploaded files by type"""
        try:
            result = await self.session.execute(
                select(UploadedFile)
                .where(
                    and_(
                        UploadedFile.video_request_id == video_request_id,
                        UploadedFile.file_type == file_type
                    )
                )
                .order_by(UploadedFile.uploaded_at)
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get files by type: {e}")
    
    async def mark_file_processed(self, file_id: str) -> bool:
        """Mark file as processed"""
        try:
            result = await self.session.execute(
                update(UploadedFile)
                .where(UploadedFile.id == file_id)
                .values(is_processed=True)
            )
            await self.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to mark file processed: {e}")

class MetricsRepository(BaseRepository):
    """Metrics repository for system metrics"""
    
    async def save_metrics(self, **kwargs) -> SystemMetrics:
        """Save system metrics"""
        try:
            metrics = SystemMetrics(**kwargs)
            self.session.add(metrics)
            await self.commit()
            return metrics
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Failed to save metrics: {e}")
    
    async def get_recent_metrics(self, hours: int = 24) -> List[SystemMetrics]:
        """Get recent metrics"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            result = await self.session.execute(
                select(SystemMetrics)
                .where(SystemMetrics.timestamp >= since)
                .order_by(desc(SystemMetrics.timestamp))
            )
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to get recent metrics: {e}")
    
    async def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            result = await self.session.execute(
                select(SystemMetrics)
                .where(SystemMetrics.timestamp >= since)
                .order_by(desc(SystemMetrics.timestamp))
            )
            metrics = result.scalars().all()
            
            if not metrics:
                return {}
            
            return {
                "avg_cpu_usage": sum(m.cpu_usage or 0 for m in metrics) / len(metrics),
                "avg_memory_usage": sum(m.memory_usage or 0 for m in metrics) / len(metrics),
                "avg_cache_hit_rate": sum(m.cache_hit_rate or 0 for m in metrics) / len(metrics),
                "total_requests": sum(m.request_count or 0 for m in metrics),
                "total_errors": sum(m.error_count or 0 for m in metrics),
                "avg_response_time": sum(m.average_response_time or 0 for m in metrics) / len(metrics)
            }
        except Exception as e:
            raise DatabaseError(f"Failed to get metrics summary: {e}") 