from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import selectinload
import structlog
from ...domain.entities.user import User, UserID
from ...domain.entities.video import Video, VideoID
from ...domain.value_objects.email import Email
from ...domain.value_objects.processing_status import ProcessingStatus
from ...domain.exceptions.domain_errors import DomainNotFoundException, DomainConflictError
from .models import UserModel, VideoModel
            from datetime import datetime, timezone
        from ...domain.entities.video import Video
        from ...domain.value_objects.video_quality import VideoQuality, VideoQualityLevel
        from ...domain.value_objects.processing_status import ProcessingStatus, ProcessingStatusType
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Repository Pattern Implementations

Data access layer that abstracts database operations and maps between
domain entities and database models.
"""



logger = structlog.get_logger()


class BaseRepository(ABC):
    """Base repository with common functionality."""
    
    def __init__(self, session: AsyncSession):
        
    """__init__ function."""
self.session = session
        self._logger = structlog.get_logger(self.__class__.__name__)


class UserRepository(BaseRepository):
    """
    Repository for User entities.
    
    Handles persistence and retrieval of User domain entities.
    """
    
    async def save(self, user: User) -> None:
        """Save or update a user."""
        try:
            # Check if user exists
            existing = await self.session.get(UserModel, user.id.value)
            
            if existing:
                # Update existing user
                await self._update_user_model(existing, user)
                self._logger.debug("Updated existing user", user_id=str(user.id))
            else:
                # Create new user
                user_model = self._user_to_model(user)
                self.session.add(user_model)
                self._logger.debug("Created new user", user_id=str(user.id))
            
            await self.session.flush()
            
        except Exception as e:
            self._logger.error("Failed to save user", user_id=str(user.id), error=str(e))
            raise
    
    async def find_by_id(self, user_id: UserID) -> Optional[User]:
        """Find user by ID."""
        try:
            stmt = select(UserModel).where(
                and_(
                    UserModel.id == user_id.value,
                    UserModel.is_deleted == False
                )
            )
            result = await self.session.execute(stmt)
            user_model = result.scalar_one_or_none()
            
            if user_model:
                return self._model_to_user(user_model)
            return None
            
        except Exception as e:
            self._logger.error("Failed to find user by ID", user_id=str(user_id), error=str(e))
            raise
    
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        try:
            stmt = select(UserModel).where(
                and_(
                    UserModel.email == email.value,
                    UserModel.is_deleted == False
                )
            )
            result = await self.session.execute(stmt)
            user_model = result.scalar_one_or_none()
            
            if user_model:
                return self._model_to_user(user_model)
            return None
            
        except Exception as e:
            self._logger.error("Failed to find user by email", email=email.value, error=str(e))
            raise
    
    async def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        try:
            stmt = select(UserModel).where(
                and_(
                    UserModel.username == username,
                    UserModel.is_deleted == False
                )
            )
            result = await self.session.execute(stmt)
            user_model = result.scalar_one_or_none()
            
            if user_model:
                return self._model_to_user(user_model)
            return None
            
        except Exception as e:
            self._logger.error("Failed to find user by username", username=username, error=str(e))
            raise
    
    async def check_email_exists(self, email: Email, exclude_user_id: Optional[UserID] = None) -> bool:
        """Check if email already exists."""
        try:
            conditions = [
                UserModel.email == email.value,
                UserModel.is_deleted == False
            ]
            
            if exclude_user_id:
                conditions.append(UserModel.id != exclude_user_id.value)
            
            stmt = select(func.count(UserModel.id)).where(and_(*conditions))
            result = await self.session.execute(stmt)
            count = result.scalar()
            
            return count > 0
            
        except Exception as e:
            self._logger.error("Failed to check email exists", email=email.value, error=str(e))
            raise
    
    async def check_username_exists(self, username: str, exclude_user_id: Optional[UserID] = None) -> bool:
        """Check if username already exists."""
        try:
            conditions = [
                UserModel.username == username,
                UserModel.is_deleted == False
            ]
            
            if exclude_user_id:
                conditions.append(UserModel.id != exclude_user_id.value)
            
            stmt = select(func.count(UserModel.id)).where(and_(*conditions))
            result = await self.session.execute(stmt)
            count = result.scalar()
            
            return count > 0
            
        except Exception as e:
            self._logger.error("Failed to check username exists", username=username, error=str(e))
            raise
    
    async def find_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Find active users with pagination."""
        try:
            stmt = (
                select(UserModel)
                .where(
                    and_(
                        UserModel.is_active == True,
                        UserModel.is_deleted == False
                    )
                )
                .order_by(UserModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            
            result = await self.session.execute(stmt)
            user_models = result.scalars().all()
            
            return [self._model_to_user(model) for model in user_models]
            
        except Exception as e:
            self._logger.error("Failed to find active users", error=str(e))
            raise
    
    async def delete(self, user_id: UserID) -> bool:
        """Soft delete a user."""
        try:
            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id.value)
                .values(is_deleted=True)
            )
            
            result = await self.session.execute(stmt)
            return result.rowcount > 0
            
        except Exception as e:
            self._logger.error("Failed to delete user", user_id=str(user_id), error=str(e))
            raise
    
    def _user_to_model(self, user: User) -> UserModel:
        """Convert User entity to UserModel."""
        return UserModel(
            id=user.id.value,
            username=user.username,
            email=user.email.value if user.email else None,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user._is_active,
            is_premium=user.is_premium,
            is_suspended=user.is_suspended,
            is_deleted=user.is_deleted,
            video_credits=user.video_credits,
            max_video_duration=user.max_video_duration,
            videos_created_today=user.videos_created_today,
            last_video_created=user._last_video_created,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    def _model_to_user(self, model: UserModel) -> User:
        """Convert UserModel to User entity."""
        email = Email(model.email) if model.email else None
        
        user = User(
            id=UserID(model.id),
            username=model.username,
            email=email,
            first_name=model.first_name,
            last_name=model.last_name,
            is_active=model.is_active,
            is_premium=model.is_premium,
            is_suspended=model.is_suspended,
            video_credits=model.video_credits,
            max_video_duration=model.max_video_duration,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
        # Set private attributes that can't be set in constructor
        user._videos_created_today = model.videos_created_today
        user._last_video_created = model.last_video_created
        
        return user
    
    async def _update_user_model(self, model: UserModel, user: User) -> None:
        """Update UserModel from User entity."""
        model.username = user.username
        model.email = user.email.value if user.email else None
        model.first_name = user.first_name
        model.last_name = user.last_name
        model.is_active = user._is_active
        model.is_premium = user.is_premium
        model.is_suspended = user.is_suspended
        model.is_deleted = user.is_deleted
        model.video_credits = user.video_credits
        model.max_video_duration = user.max_video_duration
        model.videos_created_today = user.videos_created_today
        model.last_video_created = user._last_video_created
        model.updated_at = user.updated_at


class VideoRepository(BaseRepository):
    """
    Repository for Video entities.
    
    Handles persistence and retrieval of Video domain entities.
    """
    
    async def save(self, video: "Video") -> None:
        """Save or update a video."""
        try:
            # Check if video exists
            existing = await self.session.get(VideoModel, video.id.value)
            
            if existing:
                # Update existing video
                await self._update_video_model(existing, video)
                self._logger.debug("Updated existing video", video_id=str(video.id))
            else:
                # Create new video
                video_model = self._video_to_model(video)
                self.session.add(video_model)
                self._logger.debug("Created new video", video_id=str(video.id))
            
            await self.session.flush()
            
        except Exception as e:
            self._logger.error("Failed to save video", video_id=str(video.id), error=str(e))
            raise
    
    async def find_by_id(self, video_id: VideoID) -> Optional["Video"]:
        """Find video by ID."""
        try:
            stmt = select(VideoModel).where(
                and_(
                    VideoModel.id == video_id.value,
                    VideoModel.is_deleted == False
                )
            )
            result = await self.session.execute(stmt)
            video_model = result.scalar_one_or_none()
            
            if video_model:
                return self._model_to_video(video_model)
            return None
            
        except Exception as e:
            self._logger.error("Failed to find video by ID", video_id=str(video_id), error=str(e))
            raise
    
    async def find_by_user_id(self, user_id: UserID, limit: int = 50, offset: int = 0) -> List["Video"]:
        """Find videos by user ID."""
        try:
            stmt = (
                select(VideoModel)
                .where(
                    and_(
                        VideoModel.user_id == user_id.value,
                        VideoModel.is_deleted == False
                    )
                )
                .order_by(VideoModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            
            result = await self.session.execute(stmt)
            video_models = result.scalars().all()
            
            return [self._model_to_video(model) for model in video_models]
            
        except Exception as e:
            self._logger.error("Failed to find videos by user ID", user_id=str(user_id), error=str(e))
            raise
    
    async def find_by_status(self, status: str, limit: int = 100) -> List["Video"]:
        """Find videos by processing status."""
        try:
            stmt = (
                select(VideoModel)
                .where(
                    and_(
                        VideoModel.status == status,
                        VideoModel.is_deleted == False
                    )
                )
                .order_by(VideoModel.created_at.asc())
                .limit(limit)
            )
            
            result = await self.session.execute(stmt)
            video_models = result.scalars().all()
            
            return [self._model_to_video(model) for model in video_models]
            
        except Exception as e:
            self._logger.error("Failed to find videos by status", status=status, error=str(e))
            raise
    
    async def count_user_videos_today(self, user_id: UserID) -> int:
        """Count videos created by user today."""
        try:
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            
            stmt = (
                select(func.count(VideoModel.id))
                .where(
                    and_(
                        VideoModel.user_id == user_id.value,
                        VideoModel.created_at >= today_start,
                        VideoModel.is_deleted == False
                    )
                )
            )
            
            result = await self.session.execute(stmt)
            return result.scalar() or 0
            
        except Exception as e:
            self._logger.error("Failed to count user videos today", user_id=str(user_id), error=str(e))
            raise
    
    async def delete(self, video_id: VideoID) -> bool:
        """Soft delete a video."""
        try:
            stmt = (
                update(VideoModel)
                .where(VideoModel.id == video_id.value)
                .values(is_deleted=True)
            )
            
            result = await self.session.execute(stmt)
            return result.rowcount > 0
            
        except Exception as e:
            self._logger.error("Failed to delete video", video_id=str(video_id), error=str(e))
            raise
    
    def _video_to_model(self, video: "Video") -> VideoModel:
        """Convert Video entity to VideoModel."""
        return VideoModel(
            id=video.id.value,
            user_id=video.user_id.value,
            title=video.title,
            description=video.description,
            duration_seconds=video.duration_seconds,
            quality_level=video.quality.level.value,
            quality_config=video.quality.to_dict(),
            status=video.processing_status.status.value,
            progress_percentage=video.processing_status.progress_percentage,
            status_message=video.processing_status.message,
            error_details=video.processing_status.error_details,
            processing_started_at=video.processing_status.started_at,
            processing_completed_at=video.processing_status.completed_at,
            input_data=video.input_data,
            output_file_path=video.output_file_path,
            output_file_size=video.output_file_size,
            is_deleted=video.is_deleted,
            created_at=video.created_at,
            updated_at=video.updated_at
        )
    
    def _model_to_video(self, model: VideoModel) -> "Video":
        """Convert VideoModel to Video entity."""
        
        # Reconstruct video quality
        quality = VideoQuality.from_dict(model.quality_config) if model.quality_config else VideoQuality.create_preset(VideoQualityLevel(model.quality_level))
        
        # Reconstruct processing status
        processing_status = ProcessingStatus(
            status=ProcessingStatusType(model.status),
            progress_percentage=model.progress_percentage,
            message=model.status_message,
            error_details=model.error_details,
            started_at=model.processing_started_at,
            completed_at=model.processing_completed_at
        )
        
        return Video(
            id=VideoID(model.id),
            user_id=UserID(model.user_id),
            title=model.title,
            description=model.description,
            duration_seconds=model.duration_seconds,
            quality=quality,
            processing_status=processing_status,
            input_data=model.input_data,
            output_file_path=model.output_file_path,
            output_file_size=model.output_file_size,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    async def _update_video_model(self, model: VideoModel, video: "Video") -> None:
        """Update VideoModel from Video entity."""
        model.title = video.title
        model.description = video.description
        model.duration_seconds = video.duration_seconds
        model.quality_level = video.quality.level.value
        model.quality_config = video.quality.to_dict()
        model.status = video.processing_status.status.value
        model.progress_percentage = video.processing_status.progress_percentage
        model.status_message = video.processing_status.message
        model.error_details = video.processing_status.error_details
        model.processing_started_at = video.processing_status.started_at
        model.processing_completed_at = video.processing_status.completed_at
        model.input_data = video.input_data
        model.output_file_path = video.output_file_path
        model.output_file_size = video.output_file_size
        model.is_deleted = video.is_deleted
        model.updated_at = video.updated_at 