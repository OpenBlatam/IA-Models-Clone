"""
Video Optimizer Service
Optimizes and compresses videos for different use cases
"""

import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoOptimizer:
    """Optimizes video files for size and quality"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/optimized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def optimize_video(
        self,
        video_path: Path,
        quality: str = "high",
        target_size_mb: Optional[float] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Optimize video file
        
        Args:
            video_path: Path to input video
            quality: Quality preset (low, medium, high, ultra)
            target_size_mb: Target file size in MB (optional)
            output_path: Output path (optional)
            
        Returns:
            Path to optimized video
        """
        if output_path is None:
            output_path = self.output_dir / f"optimized_{video_path.stem}.mp4"
        
        quality_presets = {
            "low": {
                "crf": 28,
                "preset": "fast",
                "max_bitrate": "1M",
            },
            "medium": {
                "crf": 23,
                "preset": "medium",
                "max_bitrate": "2M",
            },
            "high": {
                "crf": 20,
                "preset": "slow",
                "max_bitrate": "5M",
            },
            "ultra": {
                "crf": 18,
                "preset": "veryslow",
                "max_bitrate": "10M",
            },
        }
        
        preset = quality_presets.get(quality, quality_presets["high"])
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-crf", str(preset["crf"]),
            "-preset", preset["preset"],
            "-maxrate", preset["max_bitrate"],
            "-bufsize", str(int(float(preset["max_bitrate"].replace("M", "")) * 2)) + "M"),
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",
            str(output_path)
        ]
        
        # If target size specified, calculate bitrate
        if target_size_mb:
            video_duration = await self._get_video_duration(video_path)
            if video_duration:
                target_bits = target_size_mb * 8 * 1024 * 1024  # Convert MB to bits
                video_bitrate = int((target_bits * 0.9) / video_duration)  # 90% for video, 10% for audio
                cmd[cmd.index("-maxrate") + 1] = f"{video_bitrate // 1000}k"
                cmd[cmd.index("-bufsize") + 1] = f"{video_bitrate // 500}k"
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Video optimization failed: {error_msg}")
                raise RuntimeError(f"Video optimization failed: {error_msg}")
            
            logger.info(f"Video optimized: {output_path}")
            return output_path
            
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Required for video optimization.")
    
    async def _get_video_duration(self, video_path: Path) -> Optional[float]:
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
        
        return None
    
    async def generate_thumbnail(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        time_offset: float = 1.0,
        width: int = 320,
        height: int = 180
    ) -> Path:
        """
        Generate thumbnail from video
        
        Args:
            video_path: Path to video file
            output_path: Output thumbnail path
            time_offset: Time offset in seconds
            width: Thumbnail width
            height: Thumbnail height
            
        Returns:
            Path to thumbnail image
        """
        if output_path is None:
            output_path = self.output_dir / f"thumbnail_{video_path.stem}.jpg"
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-ss", str(time_offset),
            "-vframes", "1",
            "-vf", f"scale={width}:{height}",
            "-q:v", "2",
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
                raise RuntimeError("Thumbnail generation failed")
            
            logger.info(f"Thumbnail generated: {output_path}")
            return output_path
            
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Required for thumbnail generation.")
    
    async def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            str(video_path)
        ]
        
        try:
            import json
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return json.loads(stdout.decode())
        except Exception as e:
            logger.warning(f"Failed to get video info: {str(e)}")
        
        return {}

