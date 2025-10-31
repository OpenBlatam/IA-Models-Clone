#!/usr/bin/env python3
"""
HeyGen AI Main System
=====================

Integrated AI video generation system that orchestrates:
- Avatar generation and management
- Voice synthesis and cloning
- Video rendering and effects
- Complete pipeline from text to video
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import core components
from core.avatar_manager import AvatarManager, AvatarGenerationConfig
from core.voice_engine import VoiceEngine, VoiceGenerationRequest
from core.video_renderer import VideoRenderer, VideoConfig, VideoEffect

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class HeyGenAIConfig:
    """Configuration for the HeyGen AI system."""
    
    # Avatar settings
    default_avatar_style: str = "realistic"
    default_avatar_quality: str = "high"
    enable_avatar_customization: bool = True
    
    # Voice settings
    default_voice_id: str = "voice_001"
    default_language: str = "en"
    enable_voice_cloning: bool = True
    
    # Video settings
    default_video_quality: str = "high"
    default_resolution: str = "1080p"
    default_fps: int = 30
    
    # Processing settings
    enable_parallel_processing: bool = True
    max_concurrent_jobs: int = 3
    enable_caching: bool = True
    cache_ttl_hours: int = 24

@dataclass
class VideoGenerationRequest:
    """Complete request for video generation."""
    
    # Text content
    script_text: str
    language: str = "en"
    
    # Avatar settings
    avatar_style: str = "realistic"
    avatar_customization: Optional[Dict[str, Any]] = None
    
    # Voice settings
    voice_id: Optional[str] = None
    voice_emotion: Optional[str] = None
    voice_speed: float = 1.0
    voice_pitch: float = 1.0
    
    # Video settings
    video_quality: str = "high"
    resolution: str = "1080p"
    background: Optional[str] = None
    effects: Optional[List[Dict[str, Any]]] = None
    
    # Output settings
    output_format: str = "mp4"
    enable_watermark: bool = False

@dataclass
class VideoGenerationResult:
    """Result of video generation."""
    
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
    
    def __init__(self):
        self.active_jobs = {}
        self.job_history = []
        self.max_concurrent_jobs = 3
    
    def create_job(self, request: VideoGenerationRequest) -> str:
        """Create a new job and return job ID."""
        job_id = str(uuid.uuid4())
        
        self.active_jobs[job_id] = {
            "status": "created",
            "start_time": None,
            "request": request,
            "created_at": time.time()
        }
        
        logger.info(f"Created job {job_id}")
        return job_id
    
    def start_job(self, job_id: str):
        """Mark job as started."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = "processing"
            self.active_jobs[job_id]["start_time"] = time.time()
            logger.info(f"Started job {job_id}")
    
    def complete_job(self, job_id: str, result: VideoGenerationResult):
        """Mark job as completed."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = "completed"
            self.active_jobs[job_id]["result"] = result
            self.active_jobs[job_id]["completed_at"] = time.time()
            
            # Move to history
            self.job_history.append(self.active_jobs[job_id])
            del self.active_jobs[job_id]
            
            logger.info(f"Completed job {job_id}")
    
    def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = "failed"
            self.active_jobs[job_id]["error"] = error_message
            self.active_jobs[job_id]["failed_at"] = time.time()
            
            # Move to history
            self.job_history.append(self.active_jobs[job_id])
            del self.active_jobs[job_id]
            
            logger.info(f"Failed job {job_id}: {error_message}")
    
    def can_start_job(self) -> bool:
        """Check if a new job can be started."""
        processing_jobs = [job for job in self.active_jobs.values() 
                          if job["status"] == "processing"]
        return len(processing_jobs) < self.max_concurrent_jobs
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        return self.active_jobs.get(job_id)
    
    def get_job_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent job history."""
        return self.job_history[-limit:] if self.job_history else []
    
    def get_active_jobs_count(self) -> int:
        """Get count of currently active jobs."""
        return len([job for job in self.active_jobs.values() 
                   if job["status"] == "processing"])

# =============================================================================
# Video Generation Pipeline Service
# =============================================================================

class VideoGenerationPipelineService:
    """Service for orchestrating the video generation pipeline."""
    
    def __init__(self, avatar_manager: AvatarManager, voice_engine: VoiceEngine, 
                 video_renderer: VideoRenderer):
        self.avatar_manager = avatar_manager
        self.voice_engine = voice_engine
        self.video_renderer = video_renderer
    
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        """
        Generate a complete video from text input.
        
        Args:
            request: Complete video generation request
            
        Returns:
            Video generation result with paths and metadata
        """
        start_time = time.time()
        
        try:
            logger.info("Starting video generation pipeline...")
            
            # Step 1: Generate or select avatar
            avatar_path = await self._generate_avatar(request)
            
            # Step 2: Generate speech from text
            audio_path = await self._generate_speech(request)
            
            # Step 3: Create avatar video with lip-sync
            avatar_video_path = await self._create_avatar_video(avatar_path, audio_path, request)
            
            # Step 4: Render final video
            final_video_path = await self._render_final_video(avatar_video_path, audio_path, request)
            
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
            
            logger.info(f"Video generation completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"Video generation failed: {error_message}")
            
            # Create error result
            result = VideoGenerationResult(
                success=False,
                processing_time=processing_time,
                error_message=error_message
            )
            
            return result
    
    async def _generate_avatar(self, request: VideoGenerationRequest) -> str:
        """Generate or select avatar based on request."""
        try:
            logger.info("Generating avatar...")
            
            # Create avatar generation config
            avatar_config = AvatarGenerationConfig(
                style=request.avatar_style,
                quality=request.video_quality,
                resolution=request.resolution,
                enable_expressions=True,
                enable_lighting=True
            )
            
            # Create avatar prompt
            avatar_prompt = self._create_avatar_prompt(request)
            
            # Generate avatar
            avatar_path = await self.avatar_manager.generate_avatar(avatar_prompt, avatar_config)
            
            logger.info(f"Avatar generated: {avatar_path}")
            return avatar_path
            
        except Exception as e:
            logger.error(f"Avatar generation failed: {e}")
            raise
    
    def _create_avatar_prompt(self, request: VideoGenerationRequest) -> str:
        """Create avatar generation prompt from request."""
        base_prompt = f"professional headshot portrait"
        
        # Add style-specific details
        if request.avatar_style == "realistic":
            base_prompt += ", photorealistic, high quality, professional"
        elif request.avatar_style == "cartoon":
            base_prompt += ", cartoon style, animated, friendly"
        elif request.avatar_style == "anime":
            base_prompt += ", anime style, Japanese animation, detailed"
        elif request.avatar_style == "artistic":
            base_prompt += ", artistic portrait, creative, painterly"
        
        # Add customization if provided
        if request.avatar_customization:
            for key, value in request.avatar_customization.items():
                base_prompt += f", {value}"
        
        return base_prompt
    
    async def _generate_speech(self, request: VideoGenerationRequest) -> str:
        """Generate speech from text using voice engine."""
        try:
            logger.info("Generating speech...")
            
            # Use default voice if none specified
            voice_id = request.voice_id or "voice_001"
            
            # Create voice generation request
            voice_request = VoiceGenerationRequest(
                text=request.script_text,
                voice_id=voice_id,
                language=request.language,
                quality=request.video_quality,
                emotion=request.voice_emotion,
                speed=request.voice_speed,
                pitch=request.voice_pitch
            )
            
            # Generate speech
            audio_path = await self.voice_engine.generate_speech(voice_request)
            
            logger.info(f"Speech generated: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise
    
    async def _create_avatar_video(self, avatar_path: str, audio_path: str, 
                                 request: VideoGenerationRequest) -> str:
        """Create avatar video with lip-sync."""
        try:
            logger.info("Creating avatar video with lip-sync...")
            
            # Create avatar generation config
            avatar_config = AvatarGenerationConfig(
                style=request.avatar_style,
                quality=request.video_quality,
                resolution=request.resolution,
                enable_lip_sync=True,
                enable_expressions=True
            )
            
            # Generate avatar video
            avatar_video_path = await self.avatar_manager.generate_avatar_video(
                avatar_path, audio_path, avatar_config
            )
            
            logger.info(f"Avatar video created: {avatar_video_path}")
            return avatar_video_path
            
        except Exception as e:
            logger.error(f"Avatar video creation failed: {e}")
            raise
    
    async def _render_final_video(self, avatar_video_path: str, audio_path: str,
                                request: VideoGenerationRequest) -> str:
        """Render final video with all effects and background."""
        try:
            logger.info("Rendering final video...")
            
            # Create video config
            video_config = VideoConfig(
                resolution=request.resolution,
                fps=30,  # Default FPS
                quality=request.video_quality,
                format=request.output_format,
                enable_effects=True,
                enable_optimization=True
            )
            
            # Convert effects to VideoEffect objects
            video_effects = self._convert_effects(request.effects) if request.effects else None
            
            # Render video
            final_video_path = await self.video_renderer.render_video(
                avatar_video_path=avatar_video_path,
                audio_path=audio_path,
                background=request.background,
                config=video_config,
                effects=video_effects
            )
            
            logger.info(f"Final video rendered: {final_video_path}")
            return final_video_path
            
        except Exception as e:
            logger.error(f"Final video rendering failed: {e}")
            raise
    
    def _convert_effects(self, effects: List[Dict[str, Any]]) -> List[VideoEffect]:
        """Convert effect dictionaries to VideoEffect objects."""
        video_effects = []
        
        for effect_dict in effects:
            effect = VideoEffect(
                name=effect_dict.get("name", ""),
                parameters=effect_dict.get("parameters", {}),
                start_time=effect_dict.get("start_time", 0.0),
                duration=effect_dict.get("duration", 0.0),
                enabled=effect_dict.get("enabled", True)
            )
            video_effects.append(effect)
        
        return video_effects

# =============================================================================
# Utility Service
# =============================================================================

class UtilityService:
    """Service for utility functions."""
    
    @staticmethod
    def create_output_directories():
        """Create necessary output directories."""
        directories = [
            "./generated_avatars",
            "./generated_audio",
            "./generated_videos",
            "./temp",
            "./cache"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Output directories created")

# =============================================================================
# Main HeyGen AI System
# =============================================================================

class HeyGenAISystem:
    """
    Main integrated system for AI video generation.
    
    This system orchestrates all components to create complete videos:
    1. Generate or select avatar
    2. Synthesize speech from text
    3. Create avatar video with lip-sync
    4. Render final video with effects
    """
    
    def __init__(self, config: Optional[HeyGenAIConfig] = None):
        """Initialize the HeyGen AI system."""
        self.config = config or HeyGenAIConfig()
        self.initialized = False
        
        # Initialize services
        self.avatar_manager = None
        self.voice_engine = None
        self.video_renderer = None
        self.job_service = JobManagementService()
        self.pipeline_service = None
        self.utility_service = UtilityService()
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing HeyGen AI System...")
            
            # Initialize avatar manager
            self.avatar_manager = AvatarManager()
            logger.info("Avatar Manager initialized")
            
            # Initialize voice engine
            self.voice_engine = VoiceEngine()
            logger.info("Voice Engine initialized")
            
            # Initialize video renderer
            self.video_renderer = VideoRenderer()
            logger.info("Video Renderer initialized")
            
            # Initialize pipeline service
            self.pipeline_service = VideoGenerationPipelineService(
                self.avatar_manager, self.voice_engine, self.video_renderer
            )
            logger.info("Pipeline Service initialized")
            
            # Create output directories
            self.utility_service.create_output_directories()
            
            self.initialized = True
            logger.info("HeyGen AI System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HeyGen AI System: {e}")
            raise
    
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        """
        Generate a complete video from text input.
        
        Args:
            request: Complete video generation request
            
        Returns:
            Video generation result with paths and metadata
        """
        if not self.initialized:
            raise RuntimeError("HeyGen AI System not initialized")
        
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
    
    async def clone_voice(self, reference_audio_path: str, voice_name: str) -> str:
        """Clone a voice from reference audio."""
        if not self.initialized:
            raise RuntimeError("HeyGen AI System not initialized")
        
        return await self.voice_engine.clone_voice(reference_audio_path, voice_name)
    
    async def generate_avatar_only(self, prompt: str, style: str = "realistic", 
                                 quality: str = "high") -> str:
        """Generate only an avatar without full video pipeline."""
        if not self.initialized:
            raise RuntimeError("HeyGen AI System not initialized")
        
        avatar_config = AvatarGenerationConfig(
            style=style,
            quality=quality,
            enable_expressions=False,
            enable_lighting=True
        )
        
        return await self.avatar_manager.generate_avatar(prompt, avatar_config)
    
    async def generate_speech_only(self, text: str, voice_id: str = None, 
                                 language: str = "en") -> str:
        """Generate only speech without full video pipeline."""
        if not self.initialized:
            raise RuntimeError("HeyGen AI System not initialized")
        
        voice_id = voice_id or self.config.default_voice_id
        
        voice_request = VoiceGenerationRequest(
            text=text,
            voice_id=voice_id,
            language=language,
            quality="high"
        )
        
        return await self.voice_engine.generate_speech(voice_request)
    
    def get_available_avatars(self) -> List[Dict[str, Any]]:
        """Get list of available avatar models."""
        if not self.initialized:
            return []
        
        return self.avatar_manager.get_available_avatars()
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voice models."""
        if not self.initialized:
            return []
        
        return self.voice_engine.get_available_voices()
    
    def get_supported_video_formats(self) -> List[str]:
        """Get list of supported video output formats."""
        if not self.initialized:
            return []
        
        return self.video_renderer.get_supported_formats()
    
    def get_quality_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available quality presets."""
        if not self.initialized:
            return {}
        
        return self.video_renderer.get_quality_presets()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        return self.job_service.get_job_status(job_id)
    
    def get_job_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent job history."""
        return self.job_service.get_job_history(limit)
    
    def get_active_jobs_count(self) -> int:
        """Get count of currently active jobs."""
        return self.job_service.get_active_jobs_count()
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the system."""
        if not self.initialized:
            return {"status": "uninitialized"}
        
        return {
            "status": "healthy",
            "system_initialized": self.initialized,
            "avatar_manager": self.avatar_manager.health_check() if self.avatar_manager else {"status": "not_initialized"},
            "voice_engine": self.voice_engine.health_check() if self.voice_engine else {"status": "not_initialized"},
            "video_renderer": self.video_renderer.health_check() if self.video_renderer else {"status": "not_initialized"},
            "active_jobs": self.get_active_jobs_count(),
            "total_jobs": len(self.job_service.job_history),
            "max_concurrent_jobs": self.job_service.max_concurrent_jobs
        }
    
    def cleanup(self):
        """Clean up system resources."""
        try:
            logger.info("Cleaning up HeyGen AI System...")
            
            # Clear job history
            self.job_service.job_history.clear()
            
            logger.info("HeyGen AI System cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

# =============================================================================
# Example Usage and Testing
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
            script_text="Hello! Welcome to our AI-powered video generation system. This is a demonstration of how we can create professional videos with AI avatars and synthetic voices.",
            language="en",
            avatar_style="realistic",
            video_quality="high",
            resolution="1080p",
            voice_emotion="friendly",
            effects=[
                {
                    "name": "fade_in",
                    "parameters": {},
                    "start_time": 0.0,
                    "duration": 1.0
                },
                {
                    "name": "text_overlay",
                    "parameters": {
                        "text": "AI Generated Video",
                        "position": (50, 50),
                        "font_size": 2,
                        "color": (255, 255, 255)
                    }
                }
            ]
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
        
        # Get available options
        print(f"Available avatars: {len(heygen_ai.get_available_avatars())}")
        print(f"Available voices: {len(heygen_ai.get_available_voices())}")
        print(f"Supported formats: {heygen_ai.get_supported_video_formats()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run example
    asyncio.run(main())
