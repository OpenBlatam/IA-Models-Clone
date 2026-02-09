"""
Watermarking Service
Adds watermarks to videos
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


class WatermarkService:
    """Adds watermarks to videos"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/watermarked")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def add_watermark(
        self,
        video_path: Path,
        watermark_text: Optional[str] = None,
        watermark_image: Optional[Path] = None,
        position: str = "bottom-right",
        opacity: float = 0.7,
        size: float = 0.1,  # 10% of video size
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Add watermark to video
        
        Args:
            video_path: Input video path
            watermark_text: Text watermark (optional)
            watermark_image: Image watermark path (optional)
            position: Position (top-left, top-right, bottom-left, bottom-right, center)
            opacity: Opacity (0.0 to 1.0)
            size: Size relative to video (0.0 to 1.0)
            output_path: Output path (optional)
            
        Returns:
            Path to watermarked video
        """
        if output_path is None:
            output_path = self.output_dir / f"watermarked_{video_path.stem}.mp4"
        
        if not watermark_text and not watermark_image:
            logger.warning("No watermark provided, returning original video")
            return video_path
        
        # Build FFmpeg filter
        if watermark_text:
            filter_complex = self._build_text_watermark_filter(
                watermark_text, position, opacity, size
            )
        else:
            filter_complex = self._build_image_watermark_filter(
                watermark_image, position, opacity, size
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
            
            if process.returncode != 0:
                logger.warning("Watermarking failed, returning original video")
                return video_path
            
            logger.info(f"Added watermark: {output_path}")
            return output_path
            
        except FileNotFoundError:
            logger.warning("FFmpeg not found, returning original video")
            return video_path
    
    def _build_text_watermark_filter(
        self,
        text: str,
        position: str,
        opacity: float,
        size: float
    ) -> str:
        """Build FFmpeg filter for text watermark"""
        # Position mapping
        positions = {
            "top-left": "x=10:y=10",
            "top-right": "x=w-tw-10:y=10",
            "bottom-left": "x=10:y=h-th-10",
            "bottom-right": "x=w-tw-10:y=h-th-10",
            "center": "x=(w-tw)/2:y=(h-th)/2",
        }
        
        pos = positions.get(position, positions["bottom-right"])
        
        # Escape text for FFmpeg
        text_escaped = text.replace(":", "\\:").replace("'", "\\'")
        
        filter_str = (
            f"drawtext=text='{text_escaped}':"
            f"fontsize=h*{size}:"
            f"fontcolor=white@0.{int(opacity*10)}:"
            f"{pos}"
        )
        
        return filter_str
    
    def _build_image_watermark_filter(
        self,
        image_path: Path,
        position: str,
        opacity: float,
        size: float
    ) -> str:
        """Build FFmpeg filter for image watermark"""
        # Position mapping
        positions = {
            "top-left": "x=10:y=10",
            "top-right": "x=W-w-10:y=10",
            "bottom-left": "x=10:y=H-h-10",
            "bottom-right": "x=W-w-10:y=H-h-10",
            "center": "x=(W-w)/2:y=(H-h)/2",
        }
        
        pos = positions.get(position, positions["bottom-right"])
        
        filter_str = (
            f"[1:v]scale=iw*{size}:ih*{size}[wm];"
            f"[0:v][wm]overlay={pos}:format=auto"
        )
        
        return filter_str


_watermark_service: Optional[WatermarkService] = None


def get_watermark_service(output_dir: Optional[str] = None) -> WatermarkService:
    """Get watermark service instance (singleton)"""
    global _watermark_service
    if _watermark_service is None:
        _watermark_service = WatermarkService(output_dir=output_dir)
    return _watermark_service

