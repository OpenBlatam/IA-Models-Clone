"""
Video Processor for Imagen Video Enhancer AI
============================================

Frame-by-frame video processing utilities.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes videos frame by frame for enhancement.
    
    Features:
    - Frame extraction
    - Frame-by-frame processing
    - Video analysis
    - Progress tracking
    """
    
    def __init__(self):
        """Initialize video processor."""
        self.supported_formats = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    
    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid, False otherwise
        """
        path = Path(video_path)
        
        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        if path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported video format: {path.suffix}")
            return False
        
        return True
    
    async def analyze_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        frame_interval: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze video properties.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze
            frame_interval: Analyze every Nth frame
            
        Returns:
            Video analysis dictionary
        """
        try:
            # Try to use opencv if available
            try:
                import cv2
                return await self._analyze_with_opencv(video_path, max_frames, frame_interval)
            except ImportError:
                logger.warning("OpenCV not available, using basic analysis")
                return await self._analyze_basic(video_path)
        except Exception as e:
            logger.error(f"Error analyzing video: {e}", exc_info=True)
            return {
                "error": str(e),
                "file_path": video_path,
                "file_size_mb": Path(video_path).stat().st_size / (1024 * 1024) if Path(video_path).exists() else 0
            }
    
    async def _analyze_basic(self, video_path: str) -> Dict[str, Any]:
        """Basic video analysis without OpenCV."""
        path = Path(video_path)
        file_size = path.stat().st_size / (1024 * 1024)  # MB
        
        return {
            "file_path": str(path),
            "file_size_mb": file_size,
            "format": path.suffix.lower(),
            "analysis_type": "basic",
            "note": "Install opencv-python for detailed analysis"
        }
    
    async def _analyze_with_opencv(
        self,
        video_path: str,
        max_frames: Optional[int],
        frame_interval: int
    ) -> Dict[str, Any]:
        """Analyze video using OpenCV."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                "error": "Could not open video file",
                "file_path": video_path
            }
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Analyze sample frames
            frames_analyzed = 0
            quality_issues = []
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_num % frame_interval == 0:
                    # Analyze frame quality
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    if laplacian_var < 100:
                        quality_issues.append({
                            "frame": frame_num,
                            "issue": "low_sharpness",
                            "value": laplacian_var
                        })
                    
                    frames_analyzed += 1
                    if max_frames and frames_analyzed >= max_frames:
                        break
                
                frame_num += 1
            
            return {
                "file_path": video_path,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height,
                "frames_analyzed": frames_analyzed,
                "quality_issues": quality_issues,
                "analysis_type": "detailed",
                "recommendations": self._generate_recommendations(quality_issues, fps, width, height)
            }
            
        finally:
            cap.release()
    
    def _generate_recommendations(
        self,
        quality_issues: List[Dict[str, Any]],
        fps: float,
        width: int,
        height: int
    ) -> List[str]:
        """Generate enhancement recommendations."""
        recommendations = []
        
        if quality_issues:
            recommendations.append("Consider denoising to improve frame quality")
            recommendations.append("Sharpening may help with low sharpness frames")
        
        if fps < 24:
            recommendations.append("Consider upscaling frame rate for smoother playback")
        
        if width < 1280 or height < 720:
            recommendations.append("Upscaling recommended for better resolution")
        
        return recommendations
    
    async def extract_sample_frames(
        self,
        video_path: str,
        num_frames: int = 5,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Extract sample frames from video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            output_dir: Output directory for frames
            
        Returns:
            List of frame file paths
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV required for frame extraction")
            return []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, frame_count // num_frames)
            
            output_path = Path(output_dir) if output_dir else Path(video_path).parent / "frames"
            output_path.mkdir(parents=True, exist_ok=True)
            
            extracted_frames = []
            frame_num = 0
            extracted = 0
            
            while extracted < num_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_path = output_path / f"frame_{extracted:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(str(frame_path))
                extracted += 1
                frame_num += frame_interval
            
            logger.info(f"Extracted {len(extracted_frames)} frames from {video_path}")
            return extracted_frames
            
        finally:
            cap.release()




