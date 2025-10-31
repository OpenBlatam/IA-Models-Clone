from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from core.config import get_config
from core.exceptions import ProcessingError, ValidationError, handle_async_exception
from core.types import VideoRequest, VideoResponse, ProcessingStatus, ProcessingTask, TaskPriority
from services.nlp_service import NLPService
from services.file_service import FileService
from services.validation_service import ValidationService
from database.connection import get_db_session
from database.repository import VideoRepository, TaskRepository, FileRepository
from cache_manager import cache
from async_processor import processor
from cdn_manager import cdn_manager
import structlog
        from video_pipeline import create_ugc_video_ad_with_langchain
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Video Service for OS Content UGC Video Generator
Handles video processing business logic with database persistence
"""



logger = structlog.get_logger("os_content.video_service")

class VideoService:
    """Video processing service with database persistence"""
    
    def __init__(self) -> Any:
        self.config = get_config()
        self.nlp_service = NLPService()
        self.file_service = FileService()
        self.validation_service = ValidationService()
    
    @handle_async_exception
    async def create_video(self, 
                          user_id: str,
                          title: str,
                          text_prompt: str,
                          image_files: Optional[List[str]] = None,
                          video_files: Optional[List[str]] = None,
                          language: str = "es",
                          duration_per_image: float = 3.0,
                          **kwargs) -> VideoResponse:
        """Create a new video processing request with database persistence"""
        
        # Validate input
        await self.validation_service.validate_video_request(
            user_id, title, text_prompt, image_files, video_files
        )
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Process uploaded files
        image_urls = []
        video_urls = []
        
        if image_files:
            image_urls = await self.file_service.process_uploaded_files(
                image_files, "image"
            )
        
        if video_files:
            video_urls = await self.file_service.process_uploaded_files(
                video_files, "video"
            )
        
        # Perform NLP analysis
        nlp_analysis = await self.nlp_service.analyze_text(text_prompt, language)
        
        # Save to database
        async for session in get_db_session():
            video_repo = VideoRepository(session)
            file_repo = FileRepository(session)
            task_repo = TaskRepository(session)
            
            # Create video request in database
            db_video_request = await video_repo.create_video_request(
                id=request_id,
                user_id=user_id,
                title=title,
                text_prompt=text_prompt,
                language=language,
                duration_per_image=duration_per_image,
                ugc_type=kwargs.get('ugc_type', 'ugc_video_ad'),
                description=kwargs.get('description'),
                nlp_analysis=nlp_analysis,
                metadata=kwargs,
                resolution_width=kwargs.get('resolution', (1080, 1920))[0],
                resolution_height=kwargs.get('resolution', (1080, 1920))[1]
            )
            
            # Save uploaded files to database
            for file_path in image_urls:
                await file_repo.create_uploaded_file(
                    video_request_id=request_id,
                    filename=Path(file_path).name,
                    original_filename=Path(file_path).name,
                    file_path=file_path,
                    file_size=Path(file_path).stat().st_size,
                    content_type="image/jpeg",
                    file_type="image",
                    checksum=await self.file_service.calculate_checksum(file_path)
                )
            
            for file_path in video_urls:
                await file_repo.create_uploaded_file(
                    video_request_id=request_id,
                    filename=Path(file_path).name,
                    original_filename=Path(file_path).name,
                    file_path=file_path,
                    file_size=Path(file_path).stat().st_size,
                    content_type="video/mp4",
                    file_type="video",
                    checksum=await self.file_service.calculate_checksum(file_path)
                )
            
            # Create processing task
            await task_repo.create_task(
                video_request_id=request_id,
                task_type="video_creation",
                priority=TaskPriority.NORMAL.value,
                timeout=self.config.processor.task_timeout,
                max_retries=3
            )
        
        # Submit for processing
        await processor.submit_task(
            self._process_video_task,
            request_id,
            priority=processor.TaskPriority.NORMAL,
            timeout=self.config.processor.task_timeout
        )
        
        # Create response
        response = VideoResponse(
            request_id=request_id,
            video_url="",
            status=ProcessingStatus.QUEUED,
            created_at=datetime.utcnow(),
            details={"message": "Video processing queued successfully"},
            progress=0.0,
            estimated_duration=len(image_urls + video_urls) * duration_per_image,
            nlp_analysis=nlp_analysis
        )
        
        logger.info(f"Video processing request created: {request_id}")
        return response
    
    @handle_async_exception
    async def get_video_status(self, request_id: str) -> VideoResponse:
        """Get video processing status from database"""
        
        async for session in get_db_session():
            video_repo = VideoRepository(session)
            task_repo = TaskRepository(session)
            
            # Get video request from database
            db_video_request = await video_repo.get_video_request_by_id(request_id)
            if not db_video_request:
                raise ValidationError(f"Request ID not found: {request_id}")
            
            # Get processing tasks
            tasks = await task_repo.get_tasks_by_video_request(request_id)
            
            # Check cache for completed video
            cache_key = f"video_result:{request_id}"
            cached_result = await cache.get(cache_key)
            
            if cached_result:
                return VideoResponse(
                    request_id=request_id,
                    video_url=cached_result["video_url"],
                    status=ProcessingStatus.COMPLETED,
                    created_at=db_video_request.created_at,
                    details=cached_result.get("details", {}),
                    progress=1.0,
                    estimated_duration=cached_result.get("estimated_duration"),
                    nlp_analysis=cached_result.get("nlp_analysis")
                )
            
            # Determine status from database
            status = ProcessingStatus(db_video_request.status)
            progress = db_video_request.progress or 0.0
            
            if db_video_request.error_message:
                status = ProcessingStatus.FAILED
                progress = 0.0
            
            return VideoResponse(
                request_id=request_id,
                video_url=db_video_request.video_url or "",
                status=status,
                created_at=db_video_request.created_at,
                details={"message": db_video_request.error_message or "Processing in progress"},
                progress=progress,
                estimated_duration=db_video_request.estimated_duration,
                nlp_analysis=db_video_request.nlp_analysis
            )
    
    @handle_async_exception
    async def _process_video_task(self, request_id: str) -> str:
        """Process video task with database updates"""
        
        async for session in get_db_session():
            video_repo = VideoRepository(session)
            task_repo = TaskRepository(session)
            
            try:
                # Get video request
                db_video_request = await video_repo.get_video_request_by_id(request_id)
                if not db_video_request:
                    raise ValidationError(f"Video request not found: {request_id}")
                
                # Update status to processing
                await video_repo.update_video_request(
                    request_id,
                    status=ProcessingStatus.PROCESSING.value,
                    started_at=datetime.utcnow(),
                    progress=0.1
                )
                
                logger.info(f"Starting video processing: {request_id}")
                
                # Validate files exist
                await self._validate_files(db_video_request)
                
                # Generate output path
                output_path = Path(self.config.storage.upload_dir) / f"ugc_{request_id}.mp4"
                
                # Process video
                video_path = await self._create_video(db_video_request, str(output_path))
                
                # Upload to CDN
                cdn_url = await cdn_manager.upload_content(
                    content_path=video_path,
                    content_id=request_id,
                    content_type="video"
                )
                
                # Update database
                await video_repo.update_video_request(
                    request_id,
                    status=ProcessingStatus.COMPLETED.value,
                    completed_at=datetime.utcnow(),
                    progress=1.0,
                    video_url=video_path,
                    cdn_url=cdn_url
                )
                
                # Mark task as completed
                tasks = await task_repo.get_tasks_by_video_request(request_id)
                for task in tasks:
                    if task.task_type == "video_creation":
                        await task_repo.mark_task_completed(
                            task.id,
                            {"video_path": video_path, "cdn_url": cdn_url}
                        )
                
                # Cache result
                cache_key = f"video_result:{request_id}"
                await cache.set(cache_key, {
                    "video_url": cdn_url,
                    "local_path": video_path,
                    "details": {"message": "Video generated successfully"},
                    "estimated_duration": db_video_request.estimated_duration,
                    "nlp_analysis": db_video_request.nlp_analysis
                }, ttl=self.config.cache.ttl)
                
                logger.info(f"Video processing completed: {request_id}")
                return cdn_url
                
            except Exception as e:
                # Update database with error
                await video_repo.update_video_request(
                    request_id,
                    status=ProcessingStatus.FAILED.value,
                    completed_at=datetime.utcnow(),
                    progress=0.0,
                    error_message=str(e)
                )
                
                # Mark task as failed
                tasks = await task_repo.get_tasks_by_video_request(request_id)
                for task in tasks:
                    if task.task_type == "video_creation":
                        await task_repo.mark_task_failed(task.id, str(e))
                
                logger.error(f"Video processing failed: {request_id} - {e}")
                raise ProcessingError(f"Video processing failed: {e}", stage="video_creation")
    
    @handle_async_exception
    async def _create_video(self, db_video_request, output_path: str) -> str:
        """Create video using video pipeline"""
        
        
        # Get file paths from database
        async for session in get_db_session():
            file_repo = FileRepository(session)
            image_files = await file_repo.get_files_by_type(db_video_request.id, "image")
            video_files = await file_repo.get_files_by_type(db_video_request.id, "video")
            
            image_paths = [f.file_path for f in image_files]
            video_paths = [f.file_path for f in video_files]
        
        video_path = await create_ugc_video_ad_with_langchain(
            image_paths=image_paths,
            video_paths=video_paths,
            text_prompt=db_video_request.text_prompt,
            output_path=output_path,
            langchain_service=None,
            duration_per_image=db_video_request.duration_per_image,
            resolution=(db_video_request.resolution_width, db_video_request.resolution_height),
            audio_path=None,
            language=db_video_request.language
        )
        
        return video_path
    
    @handle_async_exception
    async def _validate_files(self, db_video_request) -> bool:
        """Validate that all files exist"""
        
        async for session in get_db_session():
            file_repo = FileRepository(session)
            all_files = await file_repo.get_files_by_video_request(db_video_request.id)
            
            for file_record in all_files:
                if not Path(file_record.file_path).exists():
                    raise ValidationError(f"File not found: {file_record.file_path}", field="file_path")
    
    @handle_async_exception
    async def cancel_video(self, request_id: str) -> bool:
        """Cancel video processing"""
        
        async for session in get_db_session():
            video_repo = VideoRepository(session)
            task_repo = TaskRepository(session)
            
            # Get video request
            db_video_request = await video_repo.get_video_request_by_id(request_id)
            if not db_video_request:
                raise ValidationError(f"Request ID not found: {request_id}")
            
            if db_video_request.completed_at:
                raise ValidationError("Cannot cancel completed task")
            
            # Update status
            await video_repo.update_video_request(
                request_id,
                status=ProcessingStatus.CANCELLED.value,
                completed_at=datetime.utcnow()
            )
            
            # Cancel tasks
            tasks = await task_repo.get_tasks_by_video_request(request_id)
            for task in tasks:
                await task_repo.update_task(
                    task.id,
                    status="cancelled",
                    completed_at=datetime.utcnow()
                )
        
        logger.info(f"Video processing cancelled: {request_id}")
        return True
    
    @handle_async_exception
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics from database"""
        
        async for session in get_db_session():
            video_repo = VideoRepository(session)
            
            # Get counts by status
            queued_requests = await video_repo.get_video_requests_by_status(ProcessingStatus.QUEUED.value)
            processing_requests = await video_repo.get_video_requests_by_status(ProcessingStatus.PROCESSING.value)
            completed_requests = await video_repo.get_video_requests_by_status(ProcessingStatus.COMPLETED.value)
            failed_requests = await video_repo.get_video_requests_by_status(ProcessingStatus.FAILED.value)
            
            total_requests = len(queued_requests) + len(processing_requests) + len(completed_requests) + len(failed_requests)
            
            return {
                "total_requests": total_requests,
                "queued_requests": len(queued_requests),
                "processing_requests": len(processing_requests),
                "completed_requests": len(completed_requests),
                "failed_requests": len(failed_requests),
                "success_rate": (len(completed_requests) / total_requests * 100) if total_requests > 0 else 0,
                "failure_rate": (len(failed_requests) / total_requests * 100) if total_requests > 0 else 0
            }

# Global video service instance
video_service = VideoService() 