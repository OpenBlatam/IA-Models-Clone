"""
Multi-Platform Exporter for HeyGen AI
=====================================

Provides platform-specific video export capabilities with optimized
formats, resolutions, and quality settings for enterprise-grade performance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# Video processing imports
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PlatformType:
    """Platform types for video export."""
    YOUTUBE = "YOUTUBE"
    TIKTOK = "TIKTOK"
    INSTAGRAM = "INSTAGRAM"
    LINKEDIN = "LINKEDIN"
    FACEBOOK = "FACEBOOK"
    TWITTER = "TWITTER"
    TWITCH = "TWITCH"


class VideoFormat:
    """Supported video formats."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"


@dataclass
class ExportConfig:
    """Configuration for video export."""
    
    platform: PlatformType
    quality: str = "high"
    format: VideoFormat = VideoFormat.MP4
    resolution: str = "auto"
    fps: int = 30
    bitrate: str = "auto"
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlatformSpecs:
    """Platform-specific video specifications."""
    
    platform: PlatformType
    max_duration: float
    max_file_size: int
    supported_resolutions: List[str]
    supported_formats: List[str]
    aspect_ratios: List[str]
    recommended_bitrates: Dict[str, str]
    requirements: Dict[str, Any]


@dataclass
class ExportRequest:
    """Request for video export."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_path: str = ""
    platform: PlatformType = PlatformType.YOUTUBE
    config: ExportConfig = field(default_factory=lambda: ExportConfig(PlatformType.YOUTUBE))
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExportResult:
    """Result of video export."""
    
    request_id: str
    output_path: str
    platform: PlatformType
    file_size: int
    duration: float
    metadata: Dict[str, Any]
    export_time: float
    quality_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class MultiPlatformExporter(BaseService):
    """Multi-platform video export service."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-platform exporter."""
        super().__init__("MultiPlatformExporter", ServiceType.PHASE2, config)
        
        # Platform specifications
        self.platform_specs: Dict[PlatformType, PlatformSpecs] = {}
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.export_stats = {
            "total_exports": 0,
            "successful_exports": 0,
            "failed_exports": 0,
            "average_export_time": 0.0,
            "total_exported_duration": 0.0
        }
        
        # Supported platforms
        self.supported_platforms = [
            PlatformType.YOUTUBE, PlatformType.TIKTOK, PlatformType.INSTAGRAM,
            PlatformType.LINKEDIN, PlatformType.FACEBOOK, PlatformType.TWITTER,
            PlatformType.TWITCH
        ]

    async def _initialize_service_impl(self) -> None:
        """Initialize platform export services."""
        try:
            logger.info("Initializing multi-platform exporter...")
            
            # Load platform specifications
            await self._load_platform_specs()
            
            # Check dependencies
            await self._check_dependencies()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Multi-platform exporter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-platform exporter: {e}")
            raise

    async def _load_platform_specs(self) -> None:
        """Load platform-specific specifications."""
        try:
            # YouTube specifications
            self.platform_specs[PlatformType.YOUTUBE] = PlatformSpecs(
                platform=PlatformType.YOUTUBE,
                max_duration=43200.0,  # 12 hours
                max_file_size=128 * 1024 * 1024 * 1024,  # 128 GB
                supported_resolutions=["4K", "1440p", "1080p", "720p", "480p", "360p"],
                supported_formats=["mp4", "mov", "avi", "webm"],
                aspect_ratios=["16:9", "4:3", "1:1"],
                recommended_bitrates={
                    "4K": "45000k", "1440p": "16000k", "1080p": "8000k",
                    "720p": "5000k", "480p": "2500k", "360p": "1000k"
                },
                requirements={"codec": "h264", "audio_codec": "aac"}
            )
            
            # TikTok specifications
            self.platform_specs[PlatformType.TIKTOK] = PlatformSpecs(
                platform=PlatformType.TIKTOK,
                max_duration=600.0,  # 10 minutes
                max_file_size=287 * 1024 * 1024,  # 287 MB
                supported_resolutions=["1080p", "720p"],
                supported_formats=["mp4"],
                aspect_ratios=["9:16", "1:1", "16:9"],
                recommended_bitrates={"1080p": "8000k", "720p": "5000k"},
                requirements={"codec": "h264", "audio_codec": "aac"}
            )
            
            # Instagram specifications
            self.platform_specs[PlatformType.INSTAGRAM] = PlatformSpecs(
                platform=PlatformType.INSTAGRAM,
                max_duration=600.0,  # 10 minutes
                max_file_size=100 * 1024 * 1024,  # 100 MB
                supported_resolutions=["1080p", "720p"],
                supported_formats=["mp4"],
                aspect_ratios=["1:1", "4:5", "16:9"],
                recommended_bitrates={"1080p": "8000k", "720p": "5000k"},
                requirements={"codec": "h264", "audio_codec": "aac"}
            )
            
            # LinkedIn specifications
            self.platform_specs[PlatformType.LINKEDIN] = PlatformSpecs(
                platform=PlatformType.LINKEDIN,
                max_duration=600.0,  # 10 minutes
                max_file_size=200 * 1024 * 1024,  # 200 MB
                supported_resolutions=["1080p", "720p"],
                supported_formats=["mp4"],
                aspect_ratios=["16:9", "1:1"],
                recommended_bitrates={"1080p": "8000k", "720p": "5000k"},
                requirements={"codec": "h264", "audio_codec": "aac"}
            )
            
            # Facebook specifications
            self.platform_specs[PlatformType.FACEBOOK] = PlatformSpecs(
                platform=PlatformType.FACEBOOK,
                max_duration=240.0,  # 4 minutes
                max_file_size=4 * 1024 * 1024 * 1024,  # 4 GB
                supported_resolutions=["4K", "1080p", "720p"],
                supported_formats=["mp4", "mov"],
                aspect_ratios=["16:9", "1:1", "9:16"],
                recommended_bitrates={"4K": "45000k", "1080p": "8000k", "720p": "5000k"},
                requirements={"codec": "h264", "audio_codec": "aac"}
            )
            
            # Twitter specifications
            self.platform_specs[PlatformType.TWITTER] = PlatformSpecs(
                platform=PlatformType.TWITTER,
                max_duration=140.0,  # 2 minutes 20 seconds
                max_file_size=512 * 1024 * 1024,  # 512 MB
                supported_resolutions=["1080p", "720p"],
                supported_formats=["mp4"],
                aspect_ratios=["16:9", "1:1"],
                recommended_bitrates={"1080p": "8000k", "720p": "5000k"},
                requirements={"codec": "h264", "audio_codec": "aac"}
            )
            
            # Twitch specifications
            self.platform_specs[PlatformType.TWITCH] = PlatformSpecs(
                platform=PlatformType.TWITCH,
                max_duration=0.0,  # No limit for VODs
                max_file_size=0,  # No limit for VODs
                supported_resolutions=["1080p", "720p", "480p"],
                supported_formats=["mp4"],
                aspect_ratios=["16:9"],
                recommended_bitrates={"1080p": "6000k", "720p": "4500k", "480p": "2000k"},
                requirements={"codec": "h264", "audio_codec": "aac"}
            )
            
            logger.info(f"Loaded specifications for {len(self.platform_specs)} platforms")
            
        except Exception as e:
            logger.warning(f"Failed to load some platform specifications: {e}")

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not MOVIEPY_AVAILABLE:
            missing_deps.append("moviepy")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.warning("Some export features may not be available")

    async def _validate_configuration(self) -> None:
        """Validate exporter configuration."""
        if not self.platform_specs:
            raise RuntimeError("No platform specifications loaded")
        
        if not self.supported_platforms:
            raise RuntimeError("No supported platforms configured")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def export_video(self, video_path: str, config: ExportConfig) -> str:
        """Export video for the specified platform."""
        start_time = time.time()
        
        try:
            logger.info(f"Exporting video for platform {config.platform}")
            
            # Validate input
            if not Path(video_path).exists():
                raise ValueError(f"Video file not found: {video_path}")
            
            # Get platform specifications
            platform_specs = self.platform_specs.get(config.platform)
            if not platform_specs:
                raise ValueError(f"Unsupported platform: {config.platform}")
            
            # Validate video against platform requirements
            await self._validate_video_for_platform(video_path, platform_specs)
            
            # Create export configuration
            export_config = await self._create_export_config(config, platform_specs)
            
            # Export video
            output_path = await self._perform_export(video_path, export_config)
            
            # Calculate export time
            export_time = time.time() - start_time
            
            # Update statistics
            self._update_export_stats(export_time, True)
            
            logger.info(f"Video exported successfully for {config.platform} in {export_time:.2f}s")
            return output_path
            
        except Exception as e:
            self._update_export_stats(time.time() - start_time, False)
            logger.error(f"Video export failed: {e}")
            raise

    async def _validate_video_for_platform(self, video_path: str, platform_specs: PlatformSpecs) -> None:
        """Validate video against platform requirements."""
        try:
            if not MOVIEPY_AVAILABLE:
                logger.warning("MoviePy not available, skipping video validation")
                return
            
            # Load video for analysis
            video_clip = VideoFileClip(video_path)
            
            # Check duration
            if platform_specs.max_duration > 0 and video_clip.duration > platform_specs.max_duration:
                raise ValueError(f"Video duration {video_clip.duration}s exceeds platform limit {platform_specs.max_duration}s")
            
            # Check resolution
            if video_clip.size not in platform_specs.supported_resolutions:
                logger.warning(f"Video resolution {video_clip.size} not in supported list: {platform_specs.supported_resolutions}")
            
            # Check aspect ratio
            aspect_ratio = f"{video_clip.w}:{video_clip.h}"
            if aspect_ratio not in platform_specs.aspect_ratios:
                logger.warning(f"Video aspect ratio {aspect_ratio} not in supported list: {platform_specs.aspect_ratios}")
            
            video_clip.close()
            
        except Exception as e:
            logger.warning(f"Video validation failed: {e}")

    async def _create_export_config(self, config: ExportConfig, platform_specs: PlatformSpecs) -> Dict[str, Any]:
        """Create export configuration based on platform requirements."""
        export_config = {
            "codec": "h264",
            "audio_codec": "aac",
            "fps": config.fps,
            "quality": config.quality
        }
        
        # Set resolution
        if config.resolution == "auto":
            export_config["resolution"] = platform_specs.supported_resolutions[0]
        else:
            export_config["resolution"] = config.resolution
        
        # Set bitrate
        if config.bitrate == "auto":
            export_config["bitrate"] = platform_specs.recommended_bitrates.get(
                export_config["resolution"], "8000k"
            )
        else:
            export_config["bitrate"] = config.bitrate
        
        # Set format
        export_config["format"] = config.format
        
        return export_config

    async def _perform_export(self, video_path: str, export_config: Dict[str, Any]) -> str:
        """Perform the actual video export."""
        try:
            # Create output path
            output_dir = Path(f"./exports/{export_config['format']}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"exported_{int(time.time())}.{export_config['format']}"
            
            if MOVIEPY_AVAILABLE:
                # Use MoviePy for export
                video_clip = VideoFileClip(video_path)
                
                # Apply export settings
                if export_config["resolution"] != "auto":
                    width, height = map(int, export_config["resolution"].split("x"))
                    video_clip = video_clip.resize((width, height))
                
                # Export video
                video_clip.write_videofile(
                    str(output_path),
                    codec=export_config["codec"],
                    audio_codec=export_config["audio_codec"],
                    fps=export_config["fps"],
                    bitrate=export_config["bitrate"],
                    verbose=False,
                    logger=None
                )
                
                video_clip.close()
            else:
                # Fallback: copy file
                import shutil
                shutil.copy2(video_path, output_path)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Export operation failed: {e}")
            raise

    def _update_export_stats(self, export_time: float, success: bool):
        """Update export statistics."""
        self.export_stats["total_exports"] += 1
        
        if success:
            self.export_stats["successful_exports"] += 1
        else:
            self.export_stats["failed_exports"] += 1
        
        # Update average export time
        current_avg = self.export_stats["average_export_time"]
        total_successful = self.export_stats["successful_exports"]
        
        if total_successful > 0:
            self.export_stats["average_export_time"] = (
                (current_avg * (total_successful - 1) + export_time) / total_successful
            )

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the multi-platform exporter."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "moviepy": MOVIEPY_AVAILABLE
            }
            
            # Check platform specifications
            platform_health = {
                "total_platforms": len(self.platform_specs),
                "supported_platforms": len(self.supported_platforms),
                "platforms_loaded": list(self.platform_specs.keys())
            }
            
            # Update base health
            base_health.details.update({
                "dependencies": dependencies,
                "platforms": platform_health,
                "export_stats": self.export_stats
            })
            
            return base_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )

    async def get_supported_platforms(self) -> List[str]:
        """Get list of supported platforms."""
        return self.supported_platforms.copy()

    async def get_platform_specs(self, platform: PlatformType) -> Optional[PlatformSpecs]:
        """Get specifications for a specific platform."""
        return self.platform_specs.get(platform)

    async def get_export_recommendations(self, video_path: str, platform: PlatformType) -> Dict[str, Any]:
        """Get export recommendations for a video on a specific platform."""
        try:
            if not Path(video_path).exists():
                return {"error": "Video file not found"}
            
            platform_specs = self.platform_specs.get(platform)
            if not platform_specs:
                return {"error": "Platform not supported"}
            
            recommendations = {
                "platform": platform,
                "max_duration": platform_specs.max_duration,
                "max_file_size": platform_specs.max_file_size,
                "supported_resolutions": platform_specs.supported_resolutions,
                "supported_formats": platform_specs.supported_formats,
                "aspect_ratios": platform_specs.aspect_ratios,
                "recommended_bitrates": platform_specs.recommended_bitrates,
                "requirements": platform_specs.requirements
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get export recommendations: {e}")
            return {"error": str(e)}

    async def cleanup_temp_files(self) -> None:
        """Clean up temporary export files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for export_file in temp_dir.glob("exported_*"):
                    export_file.unlink()
                    logger.debug(f"Cleaned up temp file: {export_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize exporter
    exporter = MultiPlatformExporter()
    
    # Get platform specs
    youtube_specs = exporter.get_platform_specs(PlatformType.YOUTUBE)
    tiktok_specs = exporter.get_platform_specs(PlatformType.TIKTOK)
    
    print(f"YouTube max duration: {youtube_specs.max_duration}s")
    print(f"TikTok resolution: {tiktok_specs.resolution}")
    
    # Example export config
    config = ExportConfig(
        platform=PlatformType.YOUTUBE,
        output_format=VideoFormat.MP4,
        quality="high",
        include_watermark=False
    )
    
    # Export video (example)
    # result = exporter.export_video("input_video.mp4", config)
    # print(f"Export success: {result.success}")
    
    # Get export stats
    stats = exporter.get_export_stats()
    print(f"Export stats: {stats}")
    
    # Health check
    health = exporter.health_check()
    print(f"Exporter health: {health['status']}")


