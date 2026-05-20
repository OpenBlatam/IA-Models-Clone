"""
Video Quality Analyzer for Color Grading AI
============================================

Analyzes video quality metrics.
"""

import logging
import asyncio
import subprocess
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoQualityAnalyzer:
    """
    Analyzes video quality metrics.
    
    Features:
    - Resolution analysis
    - Bitrate analysis
    - Codec information
    - Frame rate analysis
    - Quality scoring
    """
    
    def __init__(self, ffprobe_path: str = "ffprobe"):
        """
        Initialize video quality analyzer.
        
        Args:
            ffprobe_path: Path to ffprobe executable
        """
        self.ffprobe_path = ffprobe_path
    
    async def analyze_quality(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video quality.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Quality analysis results
        """
        # Get video info
        info = await self._get_video_info(video_path)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(info)
        
        # Analyze bitrate
        bitrate_analysis = self._analyze_bitrate(info)
        
        # Analyze resolution
        resolution_analysis = self._analyze_resolution(info)
        
        return {
            "quality_score": quality_score,
            "bitrate_analysis": bitrate_analysis,
            "resolution_analysis": resolution_analysis,
            "codec": info.get("codec", "unknown"),
            "fps": info.get("fps", 0),
            "duration": info.get("duration", 0),
            "file_size_mb": Path(video_path).stat().st_size / (1024 * 1024) if Path(video_path).exists() else 0,
        }
    
    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe."""
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
            
            data = json.loads(stdout.decode())
            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {}
            )
            
            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                "codec": video_stream.get("codec_name", "unknown"),
                "bitrate": int(data.get("format", {}).get("bit_rate", 0)),
                "duration": float(data.get("format", {}).get("duration", 0)),
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
    
    def _calculate_quality_score(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality score."""
        score = 0.0
        factors = {}
        
        # Resolution score (0-40 points)
        width = info.get("width", 0)
        height = info.get("height", 0)
        pixels = width * height
        
        if pixels >= 3840 * 2160:  # 4K
            res_score = 40
        elif pixels >= 1920 * 1080:  # 1080p
            res_score = 30
        elif pixels >= 1280 * 720:  # 720p
            res_score = 20
        else:
            res_score = 10
        
        score += res_score
        factors["resolution"] = res_score
        
        # Bitrate score (0-30 points)
        bitrate = info.get("bitrate", 0)
        bitrate_mbps = bitrate / 1000000
        
        if bitrate_mbps >= 20:  # High bitrate
            bitrate_score = 30
        elif bitrate_mbps >= 10:
            bitrate_score = 25
        elif bitrate_mbps >= 5:
            bitrate_score = 20
        elif bitrate_mbps >= 2:
            bitrate_score = 15
        else:
            bitrate_score = 10
        
        score += bitrate_score
        factors["bitrate"] = bitrate_score
        
        # Codec score (0-20 points)
        codec = info.get("codec", "").lower()
        if codec in ["h264", "h265", "hevc"]:
            codec_score = 20
        elif codec in ["vp9", "av1"]:
            codec_score = 18
        else:
            codec_score = 15
        
        score += codec_score
        factors["codec"] = codec_score
        
        # Frame rate score (0-10 points)
        fps = info.get("fps", 0)
        if fps >= 60:
            fps_score = 10
        elif fps >= 30:
            fps_score = 8
        elif fps >= 24:
            fps_score = 6
        else:
            fps_score = 4
        
        score += fps_score
        factors["frame_rate"] = fps_score
        
        # Normalize to 0-100
        normalized_score = min(100, score)
        
        # Quality level
        if normalized_score >= 80:
            level = "excellent"
        elif normalized_score >= 60:
            level = "good"
        elif normalized_score >= 40:
            level = "fair"
        else:
            level = "poor"
        
        return {
            "score": normalized_score,
            "level": level,
            "factors": factors,
        }
    
    def _analyze_bitrate(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bitrate."""
        bitrate = info.get("bitrate", 0)
        bitrate_mbps = bitrate / 1000000
        
        # Recommended bitrates by resolution
        width = info.get("width", 0)
        height = info.get("height", 0)
        
        if width >= 3840:  # 4K
            recommended = 50.0
        elif width >= 1920:  # 1080p
            recommended = 10.0
        elif width >= 1280:  # 720p
            recommended = 5.0
        else:
            recommended = 2.0
        
        ratio = bitrate_mbps / recommended if recommended > 0 else 0
        
        return {
            "bitrate_mbps": bitrate_mbps,
            "recommended_mbps": recommended,
            "ratio": ratio,
            "status": "optimal" if 0.8 <= ratio <= 1.2 else ("high" if ratio > 1.2 else "low"),
        }
    
    def _analyze_resolution(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resolution."""
        width = info.get("width", 0)
        height = info.get("height", 0)
        aspect_ratio = width / height if height > 0 else 0
        
        # Determine standard
        if width >= 3840:
            standard = "4K UHD"
        elif width >= 1920:
            standard = "1080p Full HD"
        elif width >= 1280:
            standard = "720p HD"
        elif width >= 854:
            standard = "480p SD"
        else:
            standard = "Other"
        
        return {
            "width": width,
            "height": height,
            "pixels": width * height,
            "aspect_ratio": aspect_ratio,
            "standard": standard,
        }




