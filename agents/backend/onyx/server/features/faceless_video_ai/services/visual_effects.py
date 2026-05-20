"""
Visual Effects Service
Adds visual effects and animations to videos
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


class VisualEffectsService:
    """Adds visual effects to videos"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/effects")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def add_ken_burns_effect(
        self,
        image_path: Path,
        duration: float,
        output_path: Optional[Path] = None,
        zoom: float = 1.2,
        pan_x: float = 0.1,
        pan_y: float = 0.1
    ) -> Path:
        """
        Add Ken Burns effect (zoom and pan) to static image
        
        Args:
            image_path: Input image path
            duration: Video duration in seconds
            output_path: Output video path
            zoom: Zoom factor (1.0 = no zoom, 1.2 = 20% zoom)
            pan_x: Pan amount in X direction (0.0 to 1.0)
            pan_y: Pan amount in Y direction (0.0 to 1.0)
            
        Returns:
            Path to video with effect
        """
        if output_path is None:
            output_path = self.output_dir / f"kenburns_{image_path.stem}.mp4"
        
        fps = 30
        num_frames = int(duration * fps)
        
        # Build zoompan filter
        filter_complex = (
            f"zoompan=z='if(lte(zoom,1.0),{zoom},max(1.001,zoom-0.0015))':"
            f"d={num_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"s=1920x1080"
        )
        
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", filter_complex,
            "-t", str(duration),
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Ken Burns effect added: {output_path}")
                return output_path
            else:
                raise RuntimeError("Ken Burns effect failed")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found")
    
    async def add_fade_transitions(
        self,
        video_path: Path,
        fade_in_duration: float = 1.0,
        fade_out_duration: float = 1.0,
        output_path: Optional[Path] = None
    ) -> Path:
        """Add fade in/out to video"""
        if output_path is None:
            output_path = self.output_dir / f"faded_{video_path.stem}.mp4"
        
        # Get video duration
        duration = await self._get_video_duration(video_path)
        
        filter_complex = (
            f"fade=t=in:st=0:d={fade_in_duration},"
            f"fade=t=out:st={duration-fade_out_duration}:d={fade_out_duration}"
        )
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", filter_complex,
            "-c:a", "copy",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Fade transitions added: {output_path}")
                return output_path
            else:
                raise RuntimeError("Fade effect failed")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found")
    
    async def add_color_grading(
        self,
        video_path: Path,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        output_path: Optional[Path] = None
    ) -> Path:
        """Add color grading to video"""
        if output_path is None:
            output_path = self.output_dir / f"graded_{video_path.stem}.mp4"
        
        filter_complex = (
            f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"
        )
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", filter_complex,
            "-c:a", "copy",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Color grading added: {output_path}")
                return output_path
            else:
                raise RuntimeError("Color grading failed")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found")
    
    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration"""
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
            stdout, _ = await process.communicate()
            
            if process.returncode == 0:
                return float(stdout.decode().strip())
        except Exception:
            pass
        
        return 0.0


_visual_effects_service: Optional[VisualEffectsService] = None


def get_visual_effects_service(output_dir: Optional[str] = None) -> VisualEffectsService:
    """Get visual effects service instance (singleton)"""
    global _visual_effects_service
    if _visual_effects_service is None:
        _visual_effects_service = VisualEffectsService(output_dir=output_dir)
    return _visual_effects_service

