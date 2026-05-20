"""
Unified Processing System for Color Grading AI
===============================================

Consolidates media processing services:
- VideoProcessor (video processing)
- ImageProcessor (image processing)
- ColorAnalyzer (color analysis)
- ColorMatcher (color matching)
- VideoQualityAnalyzer (quality analysis)

Features:
- Unified interface for all media processing
- Multi-format support
- Quality analysis
- Color analysis and matching
- Batch processing support
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .video_processor import VideoProcessor
from .image_processor import ImageProcessor
from .color_analyzer import ColorAnalyzer
from .color_matcher import ColorMatcher
from .video_quality_analyzer import VideoQualityAnalyzer

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Media types."""
    VIDEO = "video"
    IMAGE = "image"
    AUTO = "auto"  # Auto-detect


@dataclass
class ProcessingResult:
    """Processing result."""
    success: bool
    media_type: MediaType
    output_path: Optional[str] = None
    analysis: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    color_params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedProcessingSystem:
    """
    Unified media processing system.
    
    Consolidates:
    - VideoProcessor: Video processing
    - ImageProcessor: Image processing
    - ColorAnalyzer: Color analysis
    - ColorMatcher: Color matching
    - VideoQualityAnalyzer: Quality analysis
    
    Features:
    - Unified interface for all media processing
    - Multi-format support
    - Quality analysis
    - Color analysis and matching
    """
    
    def __init__(
        self,
        ffmpeg_path: str = "/usr/bin/ffmpeg",
        histogram_bins: int = 256,
        color_space: str = "RGB"
    ):
        """
        Initialize unified processing system.
        
        Args:
            ffmpeg_path: FFmpeg path
            histogram_bins: Histogram bins for analysis
            color_space: Color space for analysis
        """
        self.video_processor = VideoProcessor(ffmpeg_path=ffmpeg_path)
        self.image_processor = ImageProcessor()
        self.color_analyzer = ColorAnalyzer(
            histogram_bins=histogram_bins,
            color_space=color_space
        )
        self.color_matcher = ColorMatcher()
        self.video_quality_analyzer = VideoQualityAnalyzer()
        
        logger.info("Initialized UnifiedProcessingSystem")
    
    async def process_media(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        media_type: MediaType = MediaType.AUTO,
        color_params: Optional[Dict[str, Any]] = None,
        analyze: bool = True,
        analyze_quality: bool = True
    ) -> ProcessingResult:
        """
        Process media (video or image).
        
        Args:
            input_path: Input media path
            output_path: Optional output path
            media_type: Media type (auto-detect if AUTO)
            color_params: Optional color grading parameters
            analyze: Whether to analyze colors
            analyze_quality: Whether to analyze quality
            
        Returns:
            Processing result
        """
        start_time = datetime.now()
        
        try:
            # Detect media type if needed
            if media_type == MediaType.AUTO:
                media_type = self._detect_media_type(input_path)
            
            # Process based on type
            if media_type == MediaType.VIDEO:
                if output_path:
                    await self.video_processor.process_video(
                        input_path,
                        output_path,
                        color_params or {}
                    )
                else:
                    output_path = input_path  # No processing, just analysis
            elif media_type == MediaType.IMAGE:
                if output_path:
                    await self.image_processor.process_image(
                        input_path,
                        output_path,
                        color_params or {}
                    )
                else:
                    output_path = input_path
            
            # Analyze colors
            analysis = {}
            if analyze:
                if media_type == MediaType.VIDEO:
                    analysis = await self.color_analyzer.analyze_video(input_path)
                else:
                    analysis = await self.color_analyzer.analyze_image(input_path)
            
            # Analyze quality
            quality_score = None
            if analyze_quality and media_type == MediaType.VIDEO:
                quality_result = await self.video_quality_analyzer.analyze(input_path)
                quality_score = quality_result.get("quality_score", 0.0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                media_type=media_type,
                output_path=output_path,
                analysis=analysis,
                quality_score=quality_score,
                color_params=color_params or {},
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Processing error: {e}")
            return ProcessingResult(
                success=False,
                media_type=media_type,
                error=str(e),
                execution_time=execution_time
            )
    
    async def match_colors(
        self,
        input_path: str,
        reference: Any,  # Can be path, image, or description
        media_type: MediaType = MediaType.AUTO
    ) -> Dict[str, Any]:
        """
        Match colors from reference.
        
        Args:
            input_path: Input media path
            reference: Reference (path, image, or description)
            media_type: Media type
            
        Returns:
            Color matching result
        """
        if media_type == MediaType.AUTO:
            media_type = self._detect_media_type(input_path)
        
        if media_type == MediaType.VIDEO:
            return await self.color_matcher.match_from_video(input_path, reference)
        else:
            return await self.color_matcher.match_from_image(input_path, reference)
    
    def _detect_media_type(self, path: str) -> MediaType:
        """Detect media type from file extension."""
        path_lower = path.lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if any(path_lower.endswith(ext) for ext in video_extensions):
            return MediaType.VIDEO
        elif any(path_lower.endswith(ext) for ext in image_extensions):
            return MediaType.IMAGE
        else:
            # Default to image
            return MediaType.IMAGE
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "video_processor_available": self.video_processor is not None,
            "image_processor_available": self.image_processor is not None,
            "color_analyzer_available": self.color_analyzer is not None,
            "color_matcher_available": self.color_matcher is not None,
            "quality_analyzer_available": self.video_quality_analyzer is not None,
        }


