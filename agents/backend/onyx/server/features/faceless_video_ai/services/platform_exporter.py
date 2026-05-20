"""
Platform Exporter Service
Exports videos optimized for different platforms
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


class PlatformSpec:
    """Platform specifications"""
    
    PLATFORMS = {
        "youtube": {
            "resolution": "1920x1080",
            "fps": 30,
            "max_duration": 3600,  # 1 hour
            "format": "mp4",
            "codec": "h264",
            "bitrate": "8M",
            "audio_bitrate": "192k",
        },
        "youtube_short": {
            "resolution": "1080x1920",
            "fps": 30,
            "max_duration": 60,  # 1 minute
            "format": "mp4",
            "codec": "h264",
            "bitrate": "8M",
            "audio_bitrate": "192k",
        },
        "instagram": {
            "resolution": "1080x1080",
            "fps": 30,
            "max_duration": 60,
            "format": "mp4",
            "codec": "h264",
            "bitrate": "5M",
            "audio_bitrate": "128k",
        },
        "instagram_story": {
            "resolution": "1080x1920",
            "fps": 30,
            "max_duration": 15,
            "format": "mp4",
            "codec": "h264",
            "bitrate": "5M",
            "audio_bitrate": "128k",
        },
        "instagram_reels": {
            "resolution": "1080x1920",
            "fps": 30,
            "max_duration": 90,
            "format": "mp4",
            "codec": "h264",
            "bitrate": "8M",
            "audio_bitrate": "192k",
        },
        "tiktok": {
            "resolution": "1080x1920",
            "fps": 30,
            "max_duration": 180,  # 3 minutes
            "format": "mp4",
            "codec": "h264",
            "bitrate": "8M",
            "audio_bitrate": "192k",
        },
        "facebook": {
            "resolution": "1920x1080",
            "fps": 30,
            "max_duration": 240,  # 4 minutes
            "format": "mp4",
            "codec": "h264",
            "bitrate": "4M",
            "audio_bitrate": "128k",
        },
        "twitter": {
            "resolution": "1280x720",
            "fps": 30,
            "max_duration": 140,  # 2:20
            "format": "mp4",
            "codec": "h264",
            "bitrate": "5M",
            "audio_bitrate": "128k",
        },
        "linkedin": {
            "resolution": "1920x1080",
            "fps": 30,
            "max_duration": 600,  # 10 minutes
            "format": "mp4",
            "codec": "h264",
            "bitrate": "5M",
            "audio_bitrate": "128k",
        },
    }


class PlatformExporter:
    """Exports videos for different platforms"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def export_for_platform(
        self,
        video_path: Path,
        platform: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export video optimized for platform
        
        Args:
            video_path: Input video path
            platform: Platform name
            output_path: Output path (optional)
            
        Returns:
            Path to exported video
        """
        if platform not in PlatformSpec.PLATFORMS:
            raise ValueError(f"Unknown platform: {platform}")
        
        spec = PlatformSpec.PLATFORMS[platform]
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_{platform}.mp4"
        
        # Get video duration
        duration = await self._get_video_duration(video_path)
        
        # Check if video exceeds platform limits
        if duration > spec["max_duration"]:
            logger.warning(f"Video duration ({duration}s) exceeds platform limit ({spec['max_duration']}s)")
            # Could trim video here
        
        # Build FFmpeg command
        width, height = map(int, spec["resolution"].split('x'))
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(spec["fps"]),
            "-c:v", "libx264",
            "-b:v", spec["bitrate"],
            "-c:a", "aac",
            "-b:a", spec["audio_bitrate"],
            "-movflags", "+faststart",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Platform export failed: {error_msg}")
                raise RuntimeError(f"Export failed: {error_msg}")
            
            logger.info(f"Exported video for {platform}: {output_path}")
            return output_path
            
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Required for platform export.")
    
    async def export_for_multiple_platforms(
        self,
        video_path: Path,
        platforms: List[str]
    ) -> Dict[str, Path]:
        """
        Export video for multiple platforms
        
        Args:
            video_path: Input video path
            platforms: List of platform names
            
        Returns:
            Dictionary mapping platform to output path
        """
        results = {}
        
        for platform in platforms:
            try:
                exported_path = await self.export_for_platform(video_path, platform)
                results[platform] = exported_path
            except Exception as e:
                logger.error(f"Failed to export for {platform}: {str(e)}")
                results[platform] = None
        
        return results
    
    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                duration_str = stdout.decode().strip()
                return float(duration_str)
        except Exception as e:
            logger.warning(f"Failed to get video duration: {str(e)}")
        
        return 0.0
    
    def get_platform_specs(self) -> Dict[str, Dict[str, Any]]:
        """Get all platform specifications"""
        return PlatformSpec.PLATFORMS.copy()


_platform_exporter: Optional[PlatformExporter] = None


def get_platform_exporter(output_dir: Optional[str] = None) -> PlatformExporter:
    """Get platform exporter instance (singleton)"""
    global _platform_exporter
    if _platform_exporter is None:
        _platform_exporter = PlatformExporter(output_dir=output_dir)
    return _platform_exporter

