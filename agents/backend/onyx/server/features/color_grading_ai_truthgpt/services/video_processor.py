"""
Video Processor for Color Grading AI
====================================

Handles video processing using FFmpeg with advanced color grading capabilities.
"""

import asyncio
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import shutil

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processor using FFmpeg for color grading operations.
    
    Features:
    - Extract frames for analysis
    - Apply color grading filters
    - Export to multiple formats
    - Scene detection
    - Keyframe extraction
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        Initialize video processor.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._check_ffmpeg_available()
    
    def _check_ffmpeg_available(self):
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not available")
            logger.info("FFmpeg is available")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"FFmpeg may not be available: {e}")
    
    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"FFprobe error: {stderr.decode()}")
            
            info = json.loads(stdout.decode())
            
            # Extract video stream
            video_stream = next(
                (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
                {}
            )
            
            return {
                "duration": float(info.get("format", {}).get("duration", 0)),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                "codec": video_stream.get("codec_name", "unknown"),
                "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
                "format": info.get("format", {}).get("format_name", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
    
    async def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: float = 1.0,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Extract frames from video at specified interval.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            interval: Interval in seconds between frames
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            str(output_dir / "frame_%06d.jpg")
        ]
        
        if max_frames:
            cmd.extend(["-vframes", str(max_frames)])
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.warning(f"Frame extraction warnings: {stderr.decode()}")
            
            # Get extracted frames
            frames = sorted(output_dir.glob("frame_*.jpg"))
            return [str(f) for f in frames]
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    async def apply_color_grading(
        self,
        input_path: str,
        output_path: str,
        color_params: Dict[str, Any],
        codec: str = "h264",
        quality: str = "high"
    ) -> str:
        """
        Apply color grading to video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            color_params: Color grading parameters (LUT, curves, etc.)
            codec: Video codec
            quality: Quality preset
            
        Returns:
            Path to processed video
        """
        # Build FFmpeg filter complex for color grading
        filters = self._build_color_filters(color_params)
        
        # Codec settings
        codec_settings = self._get_codec_settings(codec, quality)
        
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-vf", filters,
            *codec_settings,
            "-y",  # Overwrite output
            output_path
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"Color grading failed: {stderr.decode()}")
            
            logger.info(f"Color grading applied: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error applying color grading: {e}")
            raise
    
    def _build_color_filters(self, color_params: Dict[str, Any]) -> str:
        """
        Build FFmpeg filter string from color parameters.
        
        Args:
            color_params: Color grading parameters
            
        Returns:
            FFmpeg filter string
        """
        filters = []
        
        # LUT (Look-Up Table)
        if "lut" in color_params:
            lut_path = color_params["lut"]
            filters.append(f"lut3d={lut_path}")
        
        # Color curves
        if "curves" in color_params:
            curves = color_params["curves"]
            # Build curves filter (simplified)
            filters.append("curves=preset=strong")
        
        # Color balance
        if "color_balance" in color_params:
            balance = color_params["color_balance"]
            filters.append(
                f"colorbalance=rs={balance.get('r', 0)}:"
                f"gs={balance.get('g', 0)}:"
                f"bs={balance.get('b', 0)}"
            )
        
        # Saturation
        if "saturation" in color_params:
            sat = color_params["saturation"]
            filters.append(f"eq=saturation={sat}")
        
        # Contrast and brightness
        if "contrast" in color_params or "brightness" in color_params:
            contrast = color_params.get("contrast", 1.0)
            brightness = color_params.get("brightness", 0.0)
            filters.append(f"eq=contrast={contrast}:brightness={brightness}")
        
        # Combine filters
        return ",".join(filters) if filters else "null"
    
    def _get_codec_settings(self, codec: str, quality: str) -> List[str]:
        """
        Get codec settings based on codec and quality.
        
        Args:
            codec: Video codec
            quality: Quality preset
            
        Returns:
            List of FFmpeg codec arguments
        """
        settings = []
        
        if codec == "h264":
            settings.extend(["-c:v", "libx264"])
            if quality == "high":
                settings.extend(["-crf", "18", "-preset", "slow"])
            elif quality == "medium":
                settings.extend(["-crf", "23", "-preset", "medium"])
            else:
                settings.extend(["-crf", "28", "-preset", "fast"])
        elif codec == "h265":
            settings.extend(["-c:v", "libx265"])
            if quality == "high":
                settings.extend(["-crf", "20", "-preset", "slow"])
            else:
                settings.extend(["-crf", "26", "-preset", "medium"])
        elif codec == "prores":
            settings.extend(["-c:v", "prores_ks"])
            settings.extend(["-profile:v", "3"])  # ProRes 422 HQ
        
        return settings
    
    async def create_preview(
        self,
        video_path: str,
        output_path: str,
        duration: float = 10.0,
        resolution: str = "720p"
    ) -> str:
        """
        Create a preview clip of the video.
        
        Args:
            video_path: Input video path
            output_path: Output preview path
            duration: Preview duration in seconds
            resolution: Preview resolution
            
        Returns:
            Path to preview video
        """
        # Resolution mapping
        res_map = {
            "720p": "1280:720",
            "1080p": "1920:1080",
            "4K": "3840:2160"
        }
        scale = res_map.get(resolution, "1280:720")
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-t", str(duration),
            "-vf", f"scale={scale}",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            "-y",
            output_path
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"Preview creation failed: {stderr.decode()}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error creating preview: {e}")
            raise




