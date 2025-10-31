"""
Enhanced HeyGen AI System - Main Orchestrator
Integrates all components including new gesture/emotion control and multi-platform export.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .avatar_manager import AvatarManager
from .voice_engine import VoiceEngine
from .video_renderer import VideoRenderer
from .script_generator import ScriptGenerator
from ..data.avatar_library.avatar_library_service import AvatarLibraryService
from ..data.voice_library.voice_library_service import VoiceLibraryService
from ..data.video_templates.video_template_service import VideoTemplateService
from .gesture_emotion_controller import GestureEmotionController
from .multi_platform_exporter import MultiPlatformExporter, PlatformType, ExportConfig, VideoFormat
from .external_api_integration import ExternalAPIManager
from .performance_optimizer import PerformanceOptimizer
from .real_time_collaboration import CollaborationManager
from .advanced_analytics import AdvancedAnalyticsSystem
from .enterprise_features import EnterpriseFeatures
from ..config.config_manager import ConfigurationManager, HeyGenAIConfig
from ..monitoring.logging_service import LoggingService

logger = logging.getLogger(__name__)


@dataclass
class VideoGenerationRequest:
    """Enhanced video generation request with new features"""
    text: str
    avatar_id: Optional[str] = None
    voice_id: Optional[str] = None
    template_id: Optional[str] = None
    gesture_sequence_id: Optional[str] = None
    emotion_sequence_id: Optional[str] = None
    platform: Optional[PlatformType] = None
    export_format: VideoFormat = VideoFormat.MP4
    quality: str = "high"
    include_watermark: bool = False
    include_subtitles: bool = False
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGenerationResult:
    """Enhanced video generation result"""
    success: bool
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    avatar_path: Optional[str] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    export_results: List[Dict[str, Any]] = field(default_factory=list)


class JobManagementService:
    """Service for managing video generation jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_counter = 0
    
    def create_job(self, request: VideoGenerationRequest) -> str:
        """Create a new job"""
        self.job_counter += 1
        job_id = f"job_{self.job_counter}_{int(datetime.now().timestamp())}"
        
        self.jobs[job_id] = {
            "id": job_id,
            "request": request,
            "status": "created",
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        logger.info(f"Created job: {job_id}")
        return job_id
    
    def start_job(self, job_id: str):
        """Start a job"""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["started_at"] = datetime.now()
            logger.info(f"Started job: {job_id}")
    
    def complete_job(self, job_id: str, result: VideoGenerationResult):
        """Complete a job"""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["completed_at"] = datetime.now()
            self.jobs[job_id]["result"] = result
            logger.info(f"Completed job: {job_id}")
    
    def fail_job(self, job_id: str, error: str):
        """Mark a job as failed"""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["completed_at"] = datetime.now()
            self.jobs[job_id]["error"] = error
            logger.error(f"Failed job: {job_id} - {error}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        return self.jobs.get(job_id)
    
    def get_job_history(self) -> List[Dict[str, Any]]:
        """Get job history"""
        return list(self.jobs.values())
    
    def get_active_jobs_count(self) -> int:
        """Get count of active jobs"""
        return len([j for j in self.jobs.values() if j["status"] in ["created", "running"]])


class VideoGenerationPipelineService:
    """Enhanced pipeline service with gesture/emotion control and multi-platform export"""
    
    def __init__(self, avatar_manager: AvatarManager, voice_engine: VoiceEngine, 
                 video_renderer: VideoRenderer, gesture_controller: GestureEmotionController,
                 multi_platform_exporter: MultiPlatformExporter):
        self.avatar_manager = avatar_manager
        self.voice_engine = voice_engine
        self.video_renderer = video_renderer
        self.gesture_controller = gesture_controller
        self.multi_platform_exporter = multi_platform_exporter
    
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        """Generate video with enhanced features"""
        try:
            # Step 1: Generate avatar
            avatar_result = await self._generate_avatar(request)
            if not avatar_result["success"]:
                return VideoGenerationResult(
                    success=False,
                    error_message=f"Avatar generation failed: {avatar_result['error']}"
                )
            
            # Step 2: Generate speech
            speech_result = await self._generate_speech(request)
            if not speech_result["success"]:
                return VideoGenerationResult(
                    success=False,
                    error_message=f"Speech generation failed: {speech_result['error']}"
                )
            
            # Step 3: Generate avatar video with gestures and emotions
            avatar_video_result = await self._generate_avatar_video(
                request, avatar_result["avatar_path"], speech_result["audio_path"]
            )
            if not avatar_video_result["success"]:
                return VideoGenerationResult(
                    success=False,
                    error_message=f"Avatar video generation failed: {avatar_video_result['error']}"
                )
            
            # Step 4: Render final video
            final_video_result = await self._render_final_video(
                request, avatar_video_result["video_path"], speech_result["audio_path"]
            )
            if not final_video_result["success"]:
                return VideoGenerationResult(
                    success=False,
                    error_message=f"Final video rendering failed: {final_video_result['error']}"
                )
            
            # Step 5: Export for platforms if specified
            export_results = []
            if request.platform:
                export_result = await self._export_for_platform(
                    request, final_video_result["video_path"]
                )
                export_results.append(export_result)
            
            # Create result
            result = VideoGenerationResult(
                success=True,
                video_path=final_video_result["video_path"],
                audio_path=speech_result["audio_path"],
                avatar_path=avatar_result["avatar_path"],
                duration=final_video_result["duration"],
                metadata={
                    "avatar_id": request.avatar_id,
                    "voice_id": request.voice_id,
                    "template_id": request.template_id,
                    "gesture_sequence_id": request.gesture_sequence_id,
                    "emotion_sequence_id": request.emotion_sequence_id,
                    "platform": request.platform.value if request.platform else None,
                    "quality": request.quality
                },
                export_results=export_results
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Video generation pipeline failed: {e}")
            return VideoGenerationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _generate_avatar(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Generate avatar image"""
        try:
            avatar_path = await self.avatar_manager.generate_avatar(
                prompt=request.text,
                avatar_id=request.avatar_id
            )
            return {"success": True, "avatar_path": avatar_path}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_speech(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Generate speech from text"""
        try:
            audio_path = await self.voice_engine.generate_speech(
                text=request.text,
                voice_id=request.voice_id
            )
            return {"success": True, "audio_path": audio_path}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_avatar_video(self, request: VideoGenerationRequest, 
                                   avatar_path: str, audio_path: str) -> Dict[str, Any]:
        """Generate avatar video with gestures and emotions"""
        try:
            # Start gesture and emotion sequences if specified
            if request.gesture_sequence_id:
                self.gesture_controller.start_gesture_sequence(request.gesture_sequence_id)
            
            if request.emotion_sequence_id:
                self.gesture_controller.start_emotion_sequence(request.emotion_sequence_id)
            
            # Generate avatar video with lip sync
            video_path = await self.avatar_manager.generate_avatar_video(
                avatar_path=avatar_path,
                audio_path=audio_path,
                gesture_controller=self.gesture_controller
            )
            
            return {"success": True, "video_path": video_path}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _render_final_video(self, request: VideoGenerationRequest, 
                                avatar_video_path: str, audio_path: str) -> Dict[str, Any]:
        """Render final video with template and effects"""
        try:
            # Apply video template if specified
            template_config = None
            if request.template_id:
                template_service = VideoTemplateService()
                template = template_service.get_template(request.template_id)
                if template:
                    template_config = template
            
            # Render final video
            final_video_path = await self.video_renderer.render_video(
                avatar_video_path=avatar_video_path,
                audio_path=audio_path,
                template_config=template_config,
                quality=request.quality
            )
            
            # Get video duration
            duration = await self.video_renderer.get_video_duration(final_video_path)
            
            return {"success": True, "video_path": final_video_path, "duration": duration}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _export_for_platform(self, request: VideoGenerationRequest, 
                                 video_path: str) -> Dict[str, Any]:
        """Export video for specific platform"""
        try:
            export_config = ExportConfig(
                platform=request.platform,
                output_format=request.export_format,
                quality=request.quality,
                include_watermark=request.include_watermark,
                include_subtitles=request.include_subtitles,
                custom_settings=request.custom_settings
            )
            
            export_result = self.multi_platform_exporter.export_video(video_path, export_config)
            
            return {
                "platform": request.platform.value,
                "success": export_result.success,
                "output_path": export_result.output_path,
                "file_size_mb": export_result.file_size_mb,
                "error": export_result.error_message
            }
        except Exception as e:
            return {
                "platform": request.platform.value if request.platform else "unknown",
                "success": False,
                "error": str(e)
            }


class UtilityService:
    """Utility service for common operations"""
    
    def __init__(self, config: HeyGenAIConfig):
        self.config = config
    
    def create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            self.config.system.output_path,
            self.config.system.temp_path,
            self.config.system.logs_path,
            self.config.system.models_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created output directories")


class HeyGenAISystem:
    """Enhanced HeyGen AI System with all integrated components"""
    
    def __init__(self, config: Optional[HeyGenAIConfig] = None):
        self.config = config or HeyGenAIConfig()
        self.initialized = False
        
        # Core managers
        self.avatar_manager = None
        self.voice_engine = None
        self.video_renderer = None
        self.script_generator = None
        
        # New services
        self.avatar_library_service = None
        self.voice_library_service = None
        self.video_template_service = None
        self.gesture_emotion_controller = None
        self.multi_platform_exporter = None
        
        # Phase 3 services
        self.external_api_manager = None
        self.performance_optimizer = None
        
        # Phase 4 enterprise services
        self.real_time_collaboration = None
        self.advanced_analytics = None
        self.enterprise_features = None
        
        # Pipeline and utility services
        self.job_service = JobManagementService()
        self.pipeline_service = None
        self.utility_service = None
        
        # Configuration and logging
        self.config_manager = ConfigurationManager()
        self.logging_service = LoggingService()
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Enhanced HeyGen AI System...")
            
            # Initialize utility service first
            self.utility_service = UtilityService(self.config)
            self.utility_service.create_output_directories()
            
            # Initialize core managers
            self.avatar_manager = AvatarManager(self.config.avatar)
            self.voice_engine = VoiceEngine(self.config.voice)
            self.video_renderer = VideoRenderer(self.config.video)
            self.script_generator = ScriptGenerator()
            
            # Initialize new services
            self.avatar_library_service = AvatarLibraryService()
            self.voice_library_service = VoiceLibraryService()
            self.video_template_service = VideoTemplateService()
            self.gesture_emotion_controller = GestureEmotionController()
            self.multi_platform_exporter = MultiPlatformExporter()
            
            # Initialize Phase 3 services
            self.external_api_manager = ExternalAPIManager()
            self.performance_optimizer = PerformanceOptimizer()
            
            # Initialize Phase 4 enterprise services
            self.real_time_collaboration = CollaborationManager()
            self.advanced_analytics = AdvancedAnalyticsSystem()
            self.enterprise_features = EnterpriseFeatures()
            
            # Initialize pipeline service
            self.pipeline_service = VideoGenerationPipelineService(
                self.avatar_manager,
                self.voice_engine,
                self.video_renderer,
                self.gesture_emotion_controller,
                self.multi_platform_exporter
            )
            
            self.initialized = True
            logger.info("Enhanced HeyGen AI System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        """Generate video with enhanced features"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        # Create and start job
        job_id = self.job_service.create_job(request)
        self.job_service.start_job(job_id)
        
        try:
            # Generate video using pipeline
            result = await self.pipeline_service.generate_video(request)
            result.metadata["job_id"] = job_id
            
            # Complete job
            if result.success:
                self.job_service.complete_job(job_id, result)
            else:
                self.job_service.fail_job(job_id, result.error_message or "Unknown error")
            
            return result
            
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            self.job_service.fail_job(job_id, error_msg)
            return VideoGenerationResult(
                success=False,
                error_message=error_msg
            )
    
    # Avatar operations
    async def generate_avatar_only(self, prompt: str, avatar_id: Optional[str] = None) -> str:
        """Generate avatar image only"""
        return await self.avatar_manager.generate_avatar(prompt, avatar_id)
    
    async def clone_voice(self, audio_path: str, voice_name: str) -> str:
        """Clone voice from audio file"""
        return await self.voice_engine.clone_voice(audio_path, voice_name)
    
    async def generate_speech_only(self, text: str, voice_id: Optional[str] = None) -> str:
        """Generate speech only"""
        return await self.voice_engine.generate_speech(text, voice_id)
    
    # Job management
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        return self.job_service.get_job_status(job_id)
    
    def get_job_history(self) -> List[Dict[str, Any]]:
        """Get job history"""
        return self.job_service.get_job_history()
    
    def get_active_jobs_count(self) -> int:
        """Get count of active jobs"""
        return self.job_service.get_active_jobs_count()
    
    # Library services
    def get_avatar_library(self) -> Dict[str, Any]:
        """Get avatar library information"""
        return self.avatar_library_service.get_all_avatars()
    
    def get_voice_library(self) -> Dict[str, Any]:
        """Get voice library information"""
        return self.voice_library_service.get_all_voices()
    
    def get_video_templates(self) -> Dict[str, Any]:
        """Get video templates"""
        return self.video_template_service.get_all_templates()
    
    # Gesture and emotion control
    def get_gesture_sequences(self) -> Dict[str, Any]:
        """Get gesture sequences"""
        return self.gesture_emotion_controller.get_all_gesture_sequences()
    
    def get_emotion_sequences(self) -> Dict[str, Any]:
        """Get emotion sequences"""
        return self.gesture_emotion_controller.get_all_emotion_sequences()
    
    # Multi-platform export
    def get_platform_specs(self) -> Dict[str, Any]:
        """Get platform specifications"""
        return self.multi_platform_exporter.get_all_platform_specs()
    
    def get_export_history(self) -> List[Dict[str, Any]]:
        """Get export history"""
        return self.multi_platform_exporter.get_export_history()
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics"""
        return self.multi_platform_exporter.get_export_stats()
    
    # Health checks
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                "system_status": "healthy",
                "initialized": self.initialized,
                "components": {},
                "errors": []
            }
            
            # Check core components
            if self.avatar_manager:
                health_status["components"]["avatar_manager"] = self.avatar_manager.health_check()
            
            if self.voice_engine:
                health_status["components"]["voice_engine"] = self.voice_engine.health_check()
            
            if self.video_renderer:
                health_status["components"]["video_renderer"] = self.video_renderer.health_check()
            
            # Check new services
            if self.avatar_library_service:
                health_status["components"]["avatar_library"] = self.avatar_library_service.health_check()
            
            if self.voice_library_service:
                health_status["components"]["voice_library"] = self.voice_library_service.health_check()
            
            if self.video_template_service:
                health_status["components"]["video_templates"] = self.video_template_service.health_check()
            
            if self.gesture_emotion_controller:
                health_status["components"]["gesture_emotion"] = self.gesture_emotion_controller.health_check()
            
            if self.multi_platform_exporter:
                health_status["components"]["multi_platform_export"] = self.multi_platform_exporter.health_check()
            
            # Check Phase 3 services
            if self.external_api_manager:
                health_status["components"]["external_api"] = await self.external_api_manager.health_check_all()
            
            if self.performance_optimizer:
                health_status["components"]["performance_optimizer"] = await self.performance_optimizer.get_performance_stats()
            
            # Check Phase 4 enterprise services
            if self.real_time_collaboration:
                health_status["components"]["real_time_collaboration"] = await self.real_time_collaboration.health_check()
            
            if self.advanced_analytics:
                health_status["components"]["advanced_analytics"] = await self.advanced_analytics.health_check()
            
            if self.enterprise_features:
                health_status["components"]["enterprise_features"] = await self.enterprise_features.health_check()
            
            # Check for any errors
            for component, status in health_status["components"].items():
                if status.get("status") == "error":
                    health_status["system_status"] = "error"
                    health_status["errors"].append(f"{component}: {status.get('error', 'Unknown error')}")
                elif status.get("status") == "warning":
                    health_status["system_status"] = "warning"
            
            return health_status
            
        except Exception as e:
            return {
                "system_status": "error",
                "error": str(e),
                "errors": [str(e)]
            }


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize system
        system = HeyGenAISystem()
        
        # Create enhanced request
        request = VideoGenerationRequest(
            text="Hello! Welcome to our enhanced HeyGen AI system with gesture control and multi-platform export capabilities.",
            avatar_id="professional_business",
            voice_id="confident_speaker",
            template_id="corporate_presentation",
            gesture_sequence_id="presentation",
            emotion_sequence_id="professional_presentation",
            platform=PlatformType.YOUTUBE,
            export_format=VideoFormat.MP4,
            quality="high",
            include_watermark=True,
            include_subtitles=True
        )
        
        # Generate video
        result = await system.generate_video(request)
        
        if result.success:
            print(f"Video generated successfully: {result.video_path}")
            print(f"Duration: {result.duration}s")
            print(f"Export results: {result.export_results}")
        else:
            print(f"Video generation failed: {result.error_message}")
        
        # Get system health
        health = await system.health_check()
        print(f"System health: {health['system_status']}")
        
        # Get statistics
        export_stats = system.get_export_stats()
        print(f"Export statistics: {export_stats}")
    
    # Run example
    asyncio.run(main())


