#!/usr/bin/env python3
"""
HeyGen AI Main System
=====================

Integrated AI video generation system that orchestrates:
- Avatar generation and management
- Voice synthesis and cloning
- Video rendering and effects
- Complete pipeline from text to video

Follows best practices for deep learning and production systems.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import core components
from core.avatar_manager import (
    AvatarManager,
    AvatarGenerationConfig,
    AvatarStyle,
    AvatarQuality,
    Resolution,
)
from core.voice_engine import (
    VoiceEngine,
    VoiceGenerationConfig,
    VoiceQuality,
    AudioFormat,
)
from core.video_renderer import (
    VideoRenderer,
    VideoConfig,
    VideoEffect,
    VideoQuality,
    VideoFormat,
    VideoCodec,
)

# Import helper utilities
from utils.enum_mappers import (
    map_avatar_style,
    map_avatar_quality,
    map_resolution,
    map_voice_quality,
    map_video_quality,
    map_video_format,
    create_avatar_config_from_strings,
    create_video_config_from_strings,
)
from utils.gpu_error_handler import handle_gpu_errors, clear_gpu_memory

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class HeyGenAIConfig:
    """Configuration for the HeyGen AI system.
    
    Attributes:
        default_avatar_style: Default avatar style
        default_avatar_quality: Default avatar quality
        enable_avatar_customization: Enable avatar customization
        default_voice_id: Default voice ID
        default_language: Default language code
        enable_voice_cloning: Enable voice cloning
        default_video_quality: Default video quality
        default_resolution: Default video resolution
        default_fps: Default frames per second
        enable_parallel_processing: Enable parallel processing
        max_concurrent_jobs: Maximum concurrent jobs
    """
    default_avatar_style: str = "realistic"
    default_avatar_quality: str = "high"
    enable_avatar_customization: bool = True
    default_voice_id: str = "voice_001"
    default_language: str = "en"
    enable_voice_cloning: bool = True
    default_video_quality: str = "high"
    default_resolution: str = "1080p"
    default_fps: int = 30
    enable_parallel_processing: bool = True
    max_concurrent_jobs: int = 3


@dataclass
class VideoGenerationRequest:
    """Complete request for video generation.
    
    Attributes:
        script_text: Text content for video
        language: Language code
        avatar_style: Avatar style
        avatar_customization: Optional avatar customization
        voice_id: Optional voice ID
        video_quality: Video quality level
        resolution: Video resolution
        background: Optional background image/video
        effects: Optional list of video effects
        output_format: Output video format
    """
    script_text: str
    language: str = "en"
    avatar_style: str = "realistic"
    avatar_customization: Optional[Dict[str, Any]] = None
    voice_id: Optional[str] = None
    video_quality: str = "high"
    resolution: str = "1080p"
    background: Optional[str] = None
    effects: Optional[List[Dict[str, Any]]] = None
    output_format: str = "mp4"


@dataclass
class VideoGenerationResult:
    """Result of video generation.
    
    Attributes:
        success: Whether generation was successful
        video_path: Path to generated video
        avatar_path: Path to generated avatar
        audio_path: Path to generated audio
        processing_time: Processing time in seconds
        error_message: Error message if failed
        metadata: Additional metadata
    """
    success: bool
    video_path: Optional[str] = None
    avatar_path: Optional[str] = None
    audio_path: Optional[str] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Job Management Service
# =============================================================================

class JobManagementService:
    """Service for managing video generation jobs."""
    
    def __init__(self, max_concurrent_jobs: int = 3):
        """Initialize job management service.
        
        Args:
            max_concurrent_jobs: Maximum concurrent jobs
        """
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_history: List[Dict[str, Any]] = []
        self.max_concurrent_jobs = max_concurrent_jobs
    
    def create_job(self, request: VideoGenerationRequest) -> str:
        """Create a new job and return job ID.
        
        Args:
            request: Video generation request
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        self.active_jobs[job_id] = {
            "status": "created",
            "start_time": None,
            "request": request,
            "created_at": time.time()
        }
        
        logger.info(f"Created job {job_id}")
        return job_id
    
    def start_job(self, job_id: str) -> None:
        """Mark job as started.
        
        Args:
            job_id: Job ID
        """
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = "processing"
            self.active_jobs[job_id]["start_time"] = time.time()
            logger.info(f"Started job {job_id}")
    
    def complete_job(self, job_id: str, result: VideoGenerationResult) -> None:
        """Mark job as completed.
        
        Args:
            job_id: Job ID
            result: Generation result
        """
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = "completed"
            self.active_jobs[job_id]["result"] = result
            self.active_jobs[job_id]["completed_at"] = time.time()
            
            # Move to history
            self.job_history.append(self.active_jobs[job_id])
            del self.active_jobs[job_id]
            
            logger.info(f"Completed job {job_id}")
    
    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed.
        
        Args:
            job_id: Job ID
            error_message: Error message
        """
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = "failed"
            self.active_jobs[job_id]["error"] = error_message
            self.active_jobs[job_id]["failed_at"] = time.time()
            
            # Move to history
            self.job_history.append(self.active_jobs[job_id])
            del self.active_jobs[job_id]
            
            logger.info(f"Failed job {job_id}: {error_message}")
    
    def can_start_job(self) -> bool:
        """Check if a new job can be started.
        
        Returns:
            True if job can be started
        """
        processing_jobs = [
            job for job in self.active_jobs.values()
            if job["status"] == "processing"
        ]
        return len(processing_jobs) < self.max_concurrent_jobs
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job status dictionary
        """
        return self.active_jobs.get(job_id)
    
    def get_job_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent job history.
        
        Args:
            limit: Maximum number of jobs to return
        
        Returns:
            List of job dictionaries
        """
        return self.job_history[-limit:] if self.job_history else []
    
    def get_active_jobs_count(self) -> int:
        """Get count of currently active jobs.
        
        Returns:
            Number of active jobs
        """
        return len([
            job for job in self.active_jobs.values()
            if job["status"] == "processing"
        ])


# =============================================================================
# Video Generation Pipeline Service
# =============================================================================

class VideoGenerationPipelineService:
    """Service for orchestrating the video generation pipeline.
    
    Features:
    - Avatar generation
    - Voice synthesis
    - Video rendering
    - Error handling
    """
    
    def __init__(
        self,
        avatar_manager: AvatarManager,
        voice_engine: VoiceEngine,
        video_renderer: VideoRenderer,
    ):
        """Initialize pipeline service.
        
        Args:
            avatar_manager: Avatar manager instance
            voice_engine: Voice engine instance
            video_renderer: Video renderer instance
        """
        self.avatar_manager = avatar_manager
        self.voice_engine = voice_engine
        self.video_renderer = video_renderer
        self.logger = logging.getLogger(f"{__name__}.VideoGenerationPipelineService")
    
    async def generate_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        """Generate a complete video from text input.
        
        Args:
            request: Complete video generation request
        
        Returns:
            Video generation result with paths and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting video generation pipeline...")
            
            # Step 1: Generate avatar
            avatar_path = await self._generate_avatar(request)
            
            # Step 2: Generate speech from text
            audio_path = await self._generate_speech(request)
            
            # Step 3: Create avatar video with lip-sync
            avatar_video_path = await self._create_avatar_video(
                avatar_path, audio_path, request
            )
            
            # Step 4: Render final video
            final_video_path = await self._render_final_video(
                avatar_video_path, audio_path, request
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = VideoGenerationResult(
                success=True,
                video_path=final_video_path,
                avatar_path=avatar_path,
                audio_path=audio_path,
                processing_time=processing_time,
                metadata={
                    "avatar_style": request.avatar_style,
                    "voice_id": request.voice_id,
                    "video_quality": request.video_quality,
                    "resolution": request.resolution,
                    "language": request.language
                }
            )
            
            self.logger.info(
                f"Video generation completed successfully in {processing_time:.2f}s"
            )
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            self.logger.error(f"Video generation failed: {error_message}")
            
            return VideoGenerationResult(
                success=False,
                processing_time=processing_time,
                error_message=error_message
            )
    
    async def _generate_avatar(
        self,
        request: VideoGenerationRequest,
    ) -> str:
        """Generate or select avatar based on request.
        
        Args:
            request: Video generation request
        
        Returns:
            Path to generated avatar
        """
        try:
            self.logger.info("Generating avatar...")
            
            # Create avatar generation config using helper functions
            avatar_config = AvatarGenerationConfig(
                style=map_avatar_style(request.avatar_style),
                quality=map_avatar_quality(request.video_quality),
                resolution=map_resolution(request.resolution),
                enable_expressions=True,
            )
            
            # Create avatar prompt
            avatar_prompt = self._create_avatar_prompt(request)
            
            # Generate avatar
            avatar_path = await self.avatar_manager.generate_avatar(
                avatar_prompt, avatar_config
            )
            
            self.logger.info(f"Avatar generated: {avatar_path}")
            return avatar_path
            
        except Exception as e:
            self.logger.error(f"Avatar generation failed: {e}")
            raise
    
    def _create_avatar_prompt(
        self,
        request: VideoGenerationRequest,
    ) -> str:
        """Create avatar generation prompt from request.
        
        Args:
            request: Video generation request
        
        Returns:
            Avatar prompt string
        """
        base_prompt = "professional headshot portrait"
        
        # Add style-specific details
        style_prompts = {
            "realistic": "photorealistic, high quality, professional",
            "cartoon": "cartoon style, animated, friendly",
            "anime": "anime style, Japanese animation, detailed",
            "artistic": "artistic portrait, creative, painterly",
        }
        
        base_prompt += f", {style_prompts.get(request.avatar_style, style_prompts['realistic'])}"
        
        # Add customization if provided
        if request.avatar_customization:
            for key, value in request.avatar_customization.items():
                base_prompt += f", {value}"
        
        return base_prompt
    
    async def _generate_speech(
        self,
        request: VideoGenerationRequest,
    ) -> str:
        """Generate speech from text using voice engine.
        
        Args:
            request: Video generation request
        
        Returns:
            Path to generated audio
        """
        try:
            self.logger.info("Generating speech...")
            
            # Map quality string to enum using helper function
            voice_config = VoiceGenerationConfig(
                quality=map_voice_quality(request.video_quality),
            )
            
            # Generate speech
            audio_path = await self.voice_engine.generate_voice(
                text=request.script_text,
                config=voice_config,
                language=request.language,
            )
            
            self.logger.info(f"Speech generated: {audio_path}")
            return audio_path
            
        except Exception as e:
            self.logger.error(f"Speech generation failed: {e}")
            raise
    
    async def _create_avatar_video(
        self,
        avatar_path: str,
        audio_path: str,
        request: VideoGenerationRequest,
    ) -> str:
        """Create avatar video with lip-sync.
        
        Args:
            avatar_path: Path to avatar image
            audio_path: Path to audio file
            request: Video generation request
        
        Returns:
            Path to avatar video
        """
        try:
            self.logger.info("Creating avatar video with lip-sync...")
            
            # Create avatar generation config using helper functions
            avatar_config = AvatarGenerationConfig(
                style=map_avatar_style(request.avatar_style),
                quality=map_avatar_quality(request.video_quality),
                resolution=map_resolution(request.resolution),
                enable_lip_sync=True,
                enable_expressions=True,
            )
            
            # Generate avatar video
            avatar_video_path = await self.avatar_manager.generate_avatar_video(
                avatar_path, audio_path, avatar_config
            )
            
            self.logger.info(f"Avatar video created: {avatar_video_path}")
            return avatar_video_path
            
        except Exception as e:
            self.logger.error(f"Avatar video creation failed: {e}")
            raise
    
    async def _render_final_video(
        self,
        avatar_video_path: str,
        audio_path: str,
        request: VideoGenerationRequest,
    ) -> str:
        """Render final video with all effects and background.
        
        Args:
            avatar_video_path: Path to avatar video
            audio_path: Path to audio file
            request: Video generation request
        
        Returns:
            Path to final video
        """
        try:
            self.logger.info("Rendering final video...")
            
            # Map quality and format strings to enums using helper functions
            video_config = VideoConfig(
                resolution=request.resolution,
                fps=30,
                quality=map_video_quality(request.video_quality),
                format=map_video_format(request.output_format),
                codec=VideoCodec.H264,
                enable_effects=True,
            )
            
            # Convert effects to VideoEffect objects
            video_effects = None
            if request.effects:
                video_effects = [
                    VideoEffect(
                        name=effect.get("name", ""),
                        parameters=effect.get("parameters", {}),
                        start_time=effect.get("start_time", 0.0),
                        duration=effect.get("duration", 0.0),
                        enabled=effect.get("enabled", True),
                    )
                    for effect in request.effects
                ]
            
            # Load video frames (simplified - in production would load from file)
            import cv2
            cap = cv2.VideoCapture(avatar_video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            import numpy as np
            video_frames = np.array(frames)
            
            # Render video
            final_video_path = await self.video_renderer.render_video(
                video_frames=video_frames,
                audio_path=audio_path,
                config=video_config,
                effects=video_effects,
            )
            
            self.logger.info(f"Final video rendered: {final_video_path}")
            return final_video_path
            
        except Exception as e:
            self.logger.error(f"Final video rendering failed: {e}")
            raise


# =============================================================================
# Main HeyGen AI System
# =============================================================================

class HeyGenAISystem:
    """Main integrated system for AI video generation.
    
    This system orchestrates all components to create complete videos:
    1. Generate or select avatar
    2. Synthesize speech from text
    3. Create avatar video with lip-sync
    4. Render final video with effects
    """
    
    def __init__(self, config: Optional[HeyGenAIConfig] = None):
        """Initialize the HeyGen AI system.
        
        Args:
            config: System configuration
        """
        self.config = config or HeyGenAIConfig()
        self.initialized = False
        self.logger = logging.getLogger(f"{__name__}.HeyGenAISystem")
        
        # Initialize services
        self.avatar_manager: Optional[AvatarManager] = None
        self.voice_engine: Optional[VoiceEngine] = None
        self.video_renderer: Optional[VideoRenderer] = None
        self.job_service = JobManagementService(
            max_concurrent_jobs=self.config.max_concurrent_jobs
        )
        self.pipeline_service: Optional[VideoGenerationPipelineService] = None
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing HeyGen AI System...")
            
            # Initialize avatar manager
            self.avatar_manager = AvatarManager()
            self.logger.info("Avatar Manager initialized")
            
            # Initialize voice engine
            self.voice_engine = VoiceEngine()
            self.logger.info("Voice Engine initialized")
            
            # Initialize video renderer
            self.video_renderer = VideoRenderer()
            self.logger.info("Video Renderer initialized")
            
            # Initialize pipeline service
            if self.avatar_manager and self.voice_engine and self.video_renderer:
                self.pipeline_service = VideoGenerationPipelineService(
                    self.avatar_manager,
                    self.voice_engine,
                    self.video_renderer,
                )
                self.logger.info("Pipeline Service initialized")
            
            # Create output directories
            self._create_output_directories()
            
            self.initialized = True
            self.logger.info("HeyGen AI System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HeyGen AI System: {e}")
            raise
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = [
            "./generated_avatars",
            "./generated_audio",
            "./generated_videos",
            "./temp",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Output directories created")
    
    async def generate_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        """Generate a complete video from text input.
        
        Args:
            request: Complete video generation request
        
        Returns:
            Video generation result with paths and metadata
        
        Raises:
            RuntimeError: If system not initialized or max jobs reached
        """
        if not self.initialized:
            raise RuntimeError("HeyGen AI System not initialized")
        
        if not self.pipeline_service:
            raise RuntimeError("Pipeline service not initialized")
        
        # Check if we can start a new job
        if not self.job_service.can_start_job():
            raise RuntimeError("Maximum concurrent jobs reached")
        
        # Create and start job
        job_id = self.job_service.create_job(request)
        self.job_service.start_job(job_id)
        
        try:
            # Generate video using pipeline service
            result = await self.pipeline_service.generate_video(request)
            
            # Add job ID to metadata
            result.metadata["job_id"] = job_id
            
            # Complete job
            self.job_service.complete_job(job_id, result)
            
            return result
            
        except Exception as e:
            # Fail job
            self.job_service.fail_job(job_id, str(e))
            raise
    
    async def generate_avatar_only(
        self,
        prompt: str,
        style: str = "realistic",
        quality: str = "high",
    ) -> str:
        """Generate only an avatar without full video pipeline.
        
        Args:
            prompt: Avatar description prompt
            style: Avatar style
            quality: Generation quality
        
        Returns:
            Path to generated avatar
        
        Raises:
            RuntimeError: If system not initialized
        """
        if not self.initialized or not self.avatar_manager:
            raise RuntimeError("HeyGen AI System not initialized")
        
        # Use helper functions for enum mapping
        avatar_config = AvatarGenerationConfig(
            style=map_avatar_style(style),
            quality=map_avatar_quality(quality),
            enable_expressions=False,
        )
        
        return await self.avatar_manager.generate_avatar(prompt, avatar_config)
    
    async def generate_speech_only(
        self,
        text: str,
        language: str = "en",
        quality: str = "high",
    ) -> str:
        """Generate only speech without full video pipeline.
        
        Args:
            text: Text to synthesize
            language: Language code
            quality: Audio quality
        
        Returns:
            Path to generated audio
        
        Raises:
            RuntimeError: If system not initialized
        """
        if not self.initialized or not self.voice_engine:
            raise RuntimeError("HeyGen AI System not initialized")
        
        # Use helper function for enum mapping
        voice_config = VoiceGenerationConfig(
            quality=map_voice_quality(quality),
        )
        
        return await self.voice_engine.generate_voice(
            text=text,
            config=voice_config,
            language=language,
        )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job status dictionary
        """
        return self.job_service.get_job_status(job_id)
    
    def get_job_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent job history.
        
        Args:
            limit: Maximum number of jobs to return
        
        Returns:
            List of job dictionaries
        """
        return self.job_service.get_job_history(limit)
    
    def get_active_jobs_count(self) -> int:
        """Get count of currently active jobs.
        
        Returns:
            Number of active jobs
        """
        return self.job_service.get_active_jobs_count()
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the system.
        
        Returns:
            Health status dictionary
        """
        if not self.initialized:
            return {"status": "uninitialized"}
        
        health = {
            "status": "healthy",
            "system_initialized": self.initialized,
            "active_jobs": self.get_active_jobs_count(),
            "total_jobs": len(self.job_service.job_history),
            "max_concurrent_jobs": self.job_service.max_concurrent_jobs,
        }
        
        # Add component health checks
        if self.avatar_manager:
            health["avatar_manager"] = self.avatar_manager.health_check()
        if self.voice_engine:
            health["voice_engine"] = self.voice_engine.health_check()
        if self.video_renderer:
            health["video_renderer"] = self.video_renderer.health_check()
        
        return health


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of the HeyGen AI system."""
    try:
        # Initialize system
        config = HeyGenAIConfig(
            default_avatar_style="realistic",
            default_video_quality="high",
            enable_parallel_processing=True
        )
        
        heygen_ai = HeyGenAISystem(config)
        
        # Health check
        health = heygen_ai.health_check()
        print(f"System Health: {health}")
        
        # Example video generation request
        request = VideoGenerationRequest(
            script_text=(
                "Hello! Welcome to our AI-powered video generation system. "
                "This is a demonstration of how we can create professional "
                "videos with AI avatars and synthetic voices."
            ),
            language="en",
            avatar_style="realistic",
            video_quality="high",
            resolution="1080p",
        )
        
        # Generate video
        print("Starting video generation...")
        result = await heygen_ai.generate_video(request)
        
        if result.success:
            print(f"Video generated successfully!")
            print(f"Video path: {result.video_path}")
            print(f"Processing time: {result.processing_time:.2f}s")
        else:
            print(f"Video generation failed: {result.error_message}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
