from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel
from ...core.entities import Video, Template, Avatar, Script
from ...core.entities.video import VideoStatus, ProcessingStage
from ...core.repositories import VideoRepository, TemplateRepository, AvatarRepository, ScriptRepository
from ...shared.logging import get_logger
from ...shared.metrics import record_metric
from typing import Any, List, Dict, Optional
import logging
"""
Video Use Cases
==============

Use cases for video generation and management operations.
"""




logger = get_logger(__name__)


class CreateVideoRequest(BaseModel):
    """Request model for creating a video."""
    
    title: str
    description: Optional[str] = None
    template_id: UUID
    avatar_id: UUID
    script_content: str
    user_id: UUID
    quality: str = "high"
    format: str = "mp4"
    aspect_ratio: str = "16:9"
    background_music: Optional[str] = None
    watermark: Optional[str] = None
    tags: List[str] = []
    is_public: bool = False


class CreateVideoResponse(BaseModel):
    """Response model for video creation."""
    
    video_id: UUID
    status: str
    estimated_completion: Optional[datetime] = None
    progress: float = 0.0


class CreateVideoUseCase:
    """Use case for creating a new video."""
    
    def __init__(
        self,
        video_repository: VideoRepository,
        template_repository: TemplateRepository,
        avatar_repository: AvatarRepository,
        script_repository: ScriptRepository,
    ):
        
    """__init__ function."""
self.video_repository = video_repository
        self.template_repository = template_repository
        self.avatar_repository = avatar_repository
        self.script_repository = script_repository
    
    async def execute(self, request: CreateVideoRequest) -> CreateVideoResponse:
        """Execute the create video use case."""
        try:
            # Validate template exists and is available
            template = await self.template_repository.get_by_id(request.template_id)
            if not template:
                raise ValueError("Template not found")
            
            if not template.is_available_for_user(request.user_id, is_premium_user=True):
                raise ValueError("Template not available for user")
            
            # Validate avatar exists and is available
            avatar = await self.avatar_repository.get_by_id(request.avatar_id)
            if not avatar:
                raise ValueError("Avatar not found")
            
            if not avatar.is_available_for_user(request.user_id):
                raise ValueError("Avatar not available for user")
            
            # Create script
            script = Script(
                title=f"Script for {request.title}",
                content=request.script_content,
                user_id=request.user_id,
                creator_id=request.user_id,
            )
            script = await self.script_repository.create(script)
            
            # Create video
            video = Video(
                title=request.title,
                description=request.description,
                user_id=request.user_id,
                creator_id=request.user_id,
                template_id=request.template_id,
                avatar_id=request.avatar_id,
                quality=request.quality,
                format=request.format,
                aspect_ratio=request.aspect_ratio,
                background_music=request.background_music,
                watermark=request.watermark,
                tags=request.tags,
                is_public=request.is_public,
            )
            
            video = await self.video_repository.create(video)
            
            # Start background processing
            asyncio.create_task(self._process_video_background(video.id))
            
            # Record metrics
            await record_metric("videos_created", 1, {"user_id": str(request.user_id)})
            
            return CreateVideoResponse(
                video_id=video.id,
                status=video.status.value,
                estimated_completion=datetime.utcnow() + timedelta(minutes=5),
                progress=video.get_progress_percentage(),
            )
            
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            raise
    
    async def _process_video_background(self, video_id: UUID) -> None:
        """Background processing for video generation."""
        try:
            video = await self.video_repository.get_by_id(video_id)
            if not video:
                logger.error(f"Video {video_id} not found for processing")
                return
            
            # Start processing
            video.start_processing()
            await self.video_repository.update(video)
            
            # Process each stage
            await self._process_script_generation(video)
            await self._process_avatar_creation(video)
            await self._process_image_sync(video)
            await self._process_video_composition(video)
            await self._process_final_render(video)
            
            # Complete processing
            video.complete_processing(
                video_url="https://example.com/videos/final.mp4",
                duration=45.5,
                file_size=1024000
            )
            await self.video_repository.update(video)
            
            logger.info(f"Video {video_id} processing completed")
            
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            if video:
                video.fail_processing(str(e))
                await self.video_repository.update(video)
    
    async def _process_script_generation(self, video: Video) -> None:
        """Process script generation stage."""
        await asyncio.sleep(1)  # Simulate processing
        video.set_script_content("Generated script content")
        await self.video_repository.update(video)
    
    async def _process_avatar_creation(self, video: Video) -> None:
        """Process avatar creation stage."""
        await asyncio.sleep(2)  # Simulate processing
        video.set_avatar_video("https://example.com/avatars/video.mp4")
        await self.video_repository.update(video)
    
    async def _process_image_sync(self, video: Video) -> None:
        """Process image synchronization stage."""
        await asyncio.sleep(1)  # Simulate processing
        timeline = [{"timestamp": i * 3, "image": f"image_{i}.jpg"} for i in range(5)]
        video.set_sync_timeline(timeline)
        await self.video_repository.update(video)
    
    async def _process_video_composition(self, video: Video) -> None:
        """Process video composition stage."""
        await asyncio.sleep(2)  # Simulate processing
        video.update_stage_status(ProcessingStage.VIDEO_COMPOSITION, VideoStatus.COMPLETED)
        await self.video_repository.update(video)
    
    async def _process_final_render(self, video: Video) -> None:
        """Process final rendering stage."""
        await asyncio.sleep(3)  # Simulate processing
        video.update_stage_status(ProcessingStage.FINAL_RENDER, VideoStatus.COMPLETED)
        await self.video_repository.update(video)


class GetVideoRequest(BaseModel):
    """Request model for getting a video."""
    
    video_id: UUID
    user_id: UUID


class GetVideoResponse(BaseModel):
    """Response model for video details."""
    
    video: Dict
    processing_log: List[Dict]
    progress: float


class GetVideoUseCase:
    """Use case for getting video details."""
    
    def __init__(self, video_repository: VideoRepository):
        
    """__init__ function."""
self.video_repository = video_repository
    
    async def execute(self, request: GetVideoRequest) -> GetVideoResponse:
        """Execute the get video use case."""
        video = await self.video_repository.get_by_id(request.video_id)
        if not video:
            raise ValueError("Video not found")
        
        # Check ownership
        if video.user_id != request.user_id:
            raise ValueError("Access denied")
        
        return GetVideoResponse(
            video=video.to_summary_dict(),
            processing_log=video.get_processing_log(),
            progress=video.get_progress_percentage(),
        )


class ListVideosRequest(BaseModel):
    """Request model for listing videos."""
    
    user_id: UUID
    skip: int = 0
    limit: int = 20
    status: Optional[str] = None
    search: Optional[str] = None


class ListVideosResponse(BaseModel):
    """Response model for video list."""
    
    videos: List[Dict]
    total_count: int
    has_more: bool


class ListVideosUseCase:
    """Use case for listing user videos."""
    
    def __init__(self, video_repository: VideoRepository):
        
    """__init__ function."""
self.video_repository = video_repository
    
    async def execute(self, request: ListVideosRequest) -> ListVideosResponse:
        """Execute the list videos use case."""
        filters = {"user_id": request.user_id}
        if request.status:
            filters["status"] = request.status
        
        videos = await self.video_repository.list(
            skip=request.skip,
            limit=request.limit + 1,  # Get one extra to check if there are more
            filters=filters,
            sort_by="created_at",
            sort_order="desc"
        )
        
        has_more = len(videos) > request.limit
        if has_more:
            videos = videos[:-1]  # Remove the extra item
        
        total_count = await self.video_repository.count(filters)
        
        return ListVideosResponse(
            videos=[video.to_summary_dict() for video in videos],
            total_count=total_count,
            has_more=has_more,
        )


class UpdateVideoRequest(BaseModel):
    """Request model for updating a video."""
    
    video_id: UUID
    user_id: UUID
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None


class UpdateVideoResponse(BaseModel):
    """Response model for video update."""
    
    video: Dict


class UpdateVideoUseCase:
    """Use case for updating video details."""
    
    def __init__(self, video_repository: VideoRepository):
        
    """__init__ function."""
self.video_repository = video_repository
    
    async def execute(self, request: UpdateVideoRequest) -> UpdateVideoResponse:
        """Execute the update video use case."""
        video = await self.video_repository.get_by_id(request.video_id)
        if not video:
            raise ValueError("Video not found")
        
        # Check ownership
        if video.user_id != request.user_id:
            raise ValueError("Access denied")
        
        # Update fields
        if request.title is not None:
            video.title = request.title
        if request.description is not None:
            video.description = request.description
        if request.tags is not None:
            video.tags = request.tags
        if request.is_public is not None:
            video.is_public = request.is_public
        
        video = await self.video_repository.update(video)
        
        return UpdateVideoResponse(video=video.to_summary_dict())


class DeleteVideoRequest(BaseModel):
    """Request model for deleting a video."""
    
    video_id: UUID
    user_id: UUID


class DeleteVideoResponse(BaseModel):
    """Response model for video deletion."""
    
    success: bool


class DeleteVideoUseCase:
    """Use case for deleting a video."""
    
    def __init__(self, video_repository: VideoRepository):
        
    """__init__ function."""
self.video_repository = video_repository
    
    async def execute(self, request: DeleteVideoRequest) -> DeleteVideoResponse:
        """Execute the delete video use case."""
        video = await self.video_repository.get_by_id(request.video_id)
        if not video:
            raise ValueError("Video not found")
        
        # Check ownership
        if video.user_id != request.user_id:
            raise ValueError("Access denied")
        
        # Delete video
        success = await self.video_repository.delete(request.video_id)
        
        return DeleteVideoResponse(success=success)


class ProcessVideoRequest(BaseModel):
    """Request model for processing a video."""
    
    video_id: UUID
    user_id: UUID


class ProcessVideoResponse(BaseModel):
    """Response model for video processing."""
    
    success: bool
    status: str


class ProcessVideoUseCase:
    """Use case for processing a video."""
    
    def __init__(self, video_repository: VideoRepository):
        
    """__init__ function."""
self.video_repository = video_repository
    
    async def execute(self, request: ProcessVideoRequest) -> ProcessVideoResponse:
        """Execute the process video use case."""
        video = await self.video_repository.get_by_id(request.video_id)
        if not video:
            raise ValueError("Video not found")
        
        # Check ownership
        if video.user_id != request.user_id:
            raise ValueError("Access denied")
        
        # Start processing
        video.start_processing()
        await self.video_repository.update(video)
        
        # Start background processing
        asyncio.create_task(self._process_video_background(video.id))
        
        return ProcessVideoResponse(
            success=True,
            status=video.status.value,
        )
    
    async def _process_video_background(self, video_id: UUID) -> None:
        """Background processing for video generation."""
        # Implementation similar to CreateVideoUseCase._process_video_background
        pass 