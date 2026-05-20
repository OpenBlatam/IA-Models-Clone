"""
Video Orchestrator Service
Orchestrates the complete video generation pipeline
"""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from uuid import UUID
from datetime import datetime
import logging

from ..core.models import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatus,
    GenerationProgress,
    VideoScript,
)
from .script_processor import ScriptProcessor
from .video_generator import VideoGenerator
from .audio_generator import AudioGenerator
from .subtitle_generator import SubtitleGenerator
from .video_compositor import VideoCompositor
from .video_optimizer import VideoOptimizer
from .webhook_service import WebhookService
from .transitions import TransitionService
from .analytics import get_analytics_service
from .cache import get_cache_manager
from .storage import get_storage_manager
from .realtime import get_websocket_manager
from .versioning import get_versioning_service
from .notifications import get_notification_service
from .events import get_event_bus, EventType, VideoEvent

logger = logging.getLogger(__name__)


class VideoOrchestrator:
    """Orchestrates the complete video generation pipeline"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.script_processor = ScriptProcessor()
        self.video_generator = VideoGenerator(output_dir=str(self.output_dir / "images"))
        self.audio_generator = AudioGenerator(output_dir=str(self.output_dir / "audio"))
        self.subtitle_generator = SubtitleGenerator(output_dir=str(self.output_dir / "subtitles"))
        self.video_compositor = VideoCompositor(output_dir=str(self.output_dir / "output"))
        self.video_optimizer = VideoOptimizer(output_dir=str(self.output_dir / "optimized"))
        self.webhook_service = WebhookService()
        self.transition_service = TransitionService(output_dir=str(self.output_dir / "transitions"))
        self.analytics = get_analytics_service()
        self.cache_manager = get_cache_manager()
        self.storage_manager = get_storage_manager(use_cloud=True)
        self.websocket_manager = get_websocket_manager()
        self.versioning_service = get_versioning_service()
        self.notification_service = get_notification_service()
        self.event_bus = get_event_bus()
        
        # Job storage (use database in production)
        self.jobs: Dict[UUID, VideoGenerationResponse] = {}

    async def start_generation(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResponse:
        """
        Start video generation process
        
        Args:
            request: Video generation request
            
        Returns:
            Initial response with job ID
        """
        response = VideoGenerationResponse(
            status=VideoStatus.PENDING,
            progress=GenerationProgress(
                status=VideoStatus.PENDING,
                progress=0.0,
                current_step="Initializing",
                total_steps=5,
                completed_steps=0,
            )
        )
        
        self.jobs[response.video_id] = response
        return response

    async def process_generation(
        self,
        video_id: UUID,
        request: VideoGenerationRequest
    ):
        """
        Process video generation pipeline
        
        Args:
            video_id: Video job ID
            request: Video generation request
        """
        try:
            job = self.jobs[video_id]
            job.status = VideoStatus.PROCESSING
            
            # Record analytics
            generation_start = datetime.utcnow()
            self.analytics.record_generation_start(
                str(video_id),
                request.dict()
            )
            
            # Step 1: Process script
            await self._update_progress(
                job,
                VideoStatus.PROCESSING,
                10.0,
                "Processing script",
                1,
                5
            )
            
            # Broadcast progress via WebSocket
            await self.websocket_manager.broadcast_progress(
                video_id,
                10.0,
                "processing",
                "Processing script"
            )
            
            segments = self.script_processor.process_script(request.script)
            total_duration = self.script_processor.estimate_total_duration(segments)
            
            # Step 2: Generate images
            await self._update_progress(
                job,
                VideoStatus.GENERATING_IMAGES,
                30.0,
                "Generating images",
                2,
                5
            )
            
            segments = await self.video_generator.generate_images_for_segments(
                segments=segments,
                style=request.video_config.style.value,
                resolution=request.video_config.resolution
            )
            
            # Step 3: Generate audio
            await self._update_progress(
                job,
                VideoStatus.GENERATING_AUDIO,
                50.0,
                "Generating audio",
                3,
                5
            )
            
            audio_data = await self.audio_generator.generate_audio(
                script_text=request.script.text,
                segments=segments,
                voice=request.audio_config.voice.value,
                speed=request.audio_config.speed,
                pitch=request.audio_config.pitch,
                language=request.script.language
            )
            
            # Add background music if enabled
            if request.audio_config.background_music:
                audio_data["audio_path"] = await self.audio_generator.add_background_music(
                    Path(audio_data["audio_path"]),
                    music_style=request.audio_config.music_style,
                    volume=request.audio_config.music_volume
                )
            
            # Step 4: Generate subtitles
            await self._update_progress(
                job,
                VideoStatus.ADDING_SUBTITLES,
                70.0,
                "Generating subtitles",
                4,
                5
            )
            
            subtitles = self.subtitle_generator.generate_subtitles(
                segments=segments,
                config=request.subtitle_config.dict()
            )
            
            # Step 5: Composite video
            await self._update_progress(
                job,
                VideoStatus.COMPOSITING,
                85.0,
                "Compositing video",
                5,
                5
            )
            
            image_sequence = await self.video_generator.create_image_sequence(
                segments=segments,
                fps=request.video_config.fps,
                image_duration=request.video_config.image_duration
            )
            
            output_path = self.output_dir / "output" / f"video_{video_id}.{request.output_format}"
            final_video_path = await self.video_compositor.composite_video(
                image_sequence=image_sequence,
                audio_path=audio_data["audio_path"],
                subtitles=subtitles,
                video_config=request.video_config.dict(),
                subtitle_config=request.subtitle_config.dict(),
                output_path=output_path
            )
            
            # Step 6: Optimize video (optional)
            optimized_path = final_video_path
            if request.output_quality != "ultra":
                try:
                    optimized_path = await self.video_optimizer.optimize_video(
                        video_path=final_video_path,
                        quality=request.output_quality,
                        output_path=self.output_dir / "output" / f"optimized_{video_id}.{request.output_format}"
                    )
                except Exception as e:
                    logger.warning(f"Video optimization failed, using original: {str(e)}")
            
            # Step 7: Generate thumbnail
            thumbnail_path = None
            try:
                thumbnail_path = await self.video_optimizer.generate_thumbnail(
                    video_path=optimized_path,
                    output_path=self.output_dir / "output" / f"thumbnail_{video_id}.jpg"
                )
            except Exception as e:
                logger.warning(f"Thumbnail generation failed: {str(e)}")
            
            # Step 8: Upload to cloud storage
            storage_result = self.storage_manager.upload_video(
                optimized_path,
                str(video_id)
            )
            
            thumbnail_storage_result = None
            if thumbnail_path:
                thumbnail_storage_result = self.storage_manager.upload_thumbnail(
                    thumbnail_path,
                    str(video_id)
                )
            
            # Step 9: Complete
            job.status = VideoStatus.COMPLETED
            # Use cloud URL if available, otherwise local
            job.video_url = storage_result.get("cloud_url") or str(optimized_path)
            if thumbnail_storage_result:
                job.thumbnail_url = thumbnail_storage_result.get("cloud_url") or str(thumbnail_path)
            else:
                job.thumbnail_url = str(thumbnail_path) if thumbnail_path else None
            job.duration = total_duration
            job.file_size = optimized_path.stat().st_size if optimized_path.exists() else None
            
            generation_time = (datetime.utcnow() - generation_start).total_seconds()
            
            # Record analytics
            self.analytics.record_generation_complete(
                str(video_id),
                total_duration,
                job.file_size or 0,
                generation_time
            )
            
            await self._update_progress(
                job,
                VideoStatus.COMPLETED,
                100.0,
                "Completed",
                5,
                5
            )
            
            # Send webhook notification
            await self.webhook_service.notify_completion(
                video_id=video_id,
                video_url=job.video_url,
                duration=total_duration,
                file_size=job.file_size or 0
            )
            
            # Send WebSocket notification
            await self.websocket_manager.broadcast_completion(
                video_id=video_id,
                video_url=job.video_url
            )
            
            # Create version
            self.versioning_service.create_version(
                video_id=video_id,
                video_url=job.video_url,
                config=request.dict(),
                metadata={
                    "duration": total_duration,
                    "file_size": job.file_size,
                    "generation_time": generation_time,
                }
            )
            
            # Send email/SMS notification (if user info available)
            # Note: User info would come from request metadata in production
            user_email = request.script.metadata.get("user_email") if request.script.metadata else None
            user_phone = request.script.metadata.get("user_phone") if request.script.metadata else None
            
            if user_email or user_phone:
                await self.notification_service.notify_video_complete(
                    user_email=user_email,
                    user_phone=user_phone,
                    video_id=video_id,
                    video_url=job.video_url
                )
            
            # Publish completion event
            await self.event_bus.publish(
                EventType.VIDEO_GENERATION_COMPLETED,
                VideoEvent.generation_completed(video_id, total_duration, job.file_size or 0)
            )
            
            logger.info(f"Video generation completed: {video_id} in {generation_time:.2f}s")
            
        except Exception as e:
            # Publish failure event
            await self.event_bus.publish(
                EventType.VIDEO_GENERATION_FAILED,
                VideoEvent.generation_failed(video_id, str(e))
            )
            
            logger.error(f"Error processing video generation {video_id}: {str(e)}", exc_info=True)
            job = self.jobs.get(video_id)
            if job:
                job.status = VideoStatus.FAILED
                job.error = str(e)
                
                # Record analytics
                self.analytics.record_generation_failure(str(video_id), str(e))
                
                # Send webhook notification
                await self.webhook_service.notify_failure(
                    video_id=video_id,
                    error=str(e)
                )
                
                await self._update_progress(
                    job,
                    VideoStatus.FAILED,
                    job.progress.progress,
                    f"Error: {str(e)}",
                    job.progress.completed_steps,
                    job.progress.total_steps
                )

    async def _update_progress(
        self,
        job: VideoGenerationResponse,
        status: VideoStatus,
        progress: float,
        current_step: str,
        completed_steps: int,
        total_steps: int
    ):
        """Update job progress"""
        job.status = status
        job.progress.status = status
        job.progress.progress = progress
        job.progress.current_step = current_step
        job.progress.completed_steps = completed_steps
        job.progress.total_steps = total_steps
        job.progress.message = current_step
        
        logger.debug(f"Progress update: {progress}% - {current_step}")

