"""
Configuration for Color Grading AI TruthGPT
============================================
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class OpenRouterConfig:
    """OpenRouter configuration."""
    api_key: Optional[str] = None
    model: str = "anthropic/claude-3.5-sonnet"
    timeout: float = 120.0  # Longer timeout for video processing
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TruthGPTConfig:
    """TruthGPT configuration."""
    enabled: bool = True
    endpoint: Optional[str] = None
    timeout: float = 120.0
    optimization_type: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "truthgpt_endpoint": self.endpoint,
            "timeout": self.timeout,
            "optimization_type": self.optimization_type,
        }


@dataclass
class VideoProcessingConfig:
    """Video processing configuration."""
    ffmpeg_path: Optional[str] = None
    max_resolution: str = "4K"  # 1080p, 2K, 4K, 8K
    output_format: str = "mp4"  # mp4, mov, mkv, avi
    codec: str = "h264"  # h264, h265, prores
    quality: str = "high"  # low, medium, high, lossless
    frame_rate: Optional[float] = None  # None = preserve original
    max_parallel_jobs: int = 2  # Limit parallel video processing


@dataclass
class ColorAnalysisConfig:
    """Color analysis configuration."""
    histogram_bins: int = 256
    color_space: str = "RGB"  # RGB, HSV, LAB, XYZ
    analyze_scenes: bool = True
    scene_detection_threshold: float = 0.3
    extract_keyframes: bool = True
    keyframe_interval: float = 1.0  # seconds


@dataclass
class ColorGradingConfig:
    """Configuration for Color Grading AI Agent."""
    
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    truthgpt: TruthGPTConfig = field(default_factory=TruthGPTConfig)
    video_processing: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)
    color_analysis: ColorAnalysisConfig = field(default_factory=ColorAnalysisConfig)
    max_parallel_tasks: int = 5
    output_dir: str = "color_grading_output"
    debug: bool = False
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # OpenRouter
        if not self.openrouter.api_key:
            self.openrouter.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # TruthGPT
        if self.truthgpt.enabled:
            if not self.truthgpt.endpoint:
                self.truthgpt.endpoint = os.getenv("TRUTHGPT_ENDPOINT")
        
        # FFmpeg
        if not self.video_processing.ffmpeg_path:
            self.video_processing.ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
    
    def validate(self):
        """Validate configuration."""
        if not self.openrouter.api_key:
            raise ValueError("OpenRouter API key is required")
        
        if self.truthgpt.enabled and not self.truthgpt.endpoint:
            import logging
            logging.getLogger(__name__).warning(
                "TruthGPT endpoint not configured, running without TruthGPT optimization"
            )




