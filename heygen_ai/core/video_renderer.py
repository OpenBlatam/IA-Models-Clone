#!/usr/bin/env python3
"""
Video Renderer for HeyGen AI
============================

Production-ready video rendering system.
Follows best practices for video processing and composition.

Key Features:
- Video composition and layering
- Audio-video synchronization
- Post-processing and effects
- Multiple output formats
- Quality optimization
- Proper error handling
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Third-party imports with proper error handling
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning(
        "Audio libraries not available. "
        "Install with: pip install librosa soundfile"
    )

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy not available. Install with: pip install moviepy")

logger = logging.getLogger(__name__)


# =============================================================================
# Imports from shared module
# =============================================================================

from shared import (
    VideoQuality,
    VideoFormat,
    VideoCodec,
    VideoConfig,
    VideoEffect,
)

# =============================================================================
# Imports from utility helpers
# =============================================================================

from utils.memory_manager import process_in_batches, clear_memory


# =============================================================================
# Video Processing Utilities
# =============================================================================

class VideoProcessor:
    """Utility class for video processing operations."""
    
    @staticmethod
    def load_video(video_path: str) -> Tuple[np.ndarray, int, int]:
        """Load video and return frames, fps, and frame count.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of (frames array, fps, frame_count)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            return np.array(frames), fps, frame_count
            
        except Exception as e:
            logging.error(f"Failed to load video: {e}")
            raise
    
    @staticmethod
    def resize_frame(
        frame: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize frame to target size.
        
        Args:
            frame: Input frame
            target_size: Target (width, height)
        
        Returns:
            Resized frame
        """
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def apply_effect(
        frame: np.ndarray,
        effect: VideoEffect,
        current_time: float,
    ) -> np.ndarray:
        """Apply video effect to frame.
        
        Args:
            frame: Input frame
            effect: Effect configuration
            current_time: Current time in video
        
        Returns:
            Frame with effect applied
        """
        if not effect.enabled:
            return frame
        
        if not (effect.start_time <= current_time <= effect.start_time + effect.duration):
            return frame
        
        try:
            if effect.name == "fade_in":
                alpha = (current_time - effect.start_time) / effect.duration
                return (frame * alpha).astype(np.uint8)
            
            elif effect.name == "fade_out":
                alpha = 1.0 - (current_time - effect.start_time) / effect.duration
                return (frame * alpha).astype(np.uint8)
            
            elif effect.name == "blur":
                kernel_size = effect.parameters.get("kernel_size", 15)
                return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            
            elif effect.name == "brightness":
                factor = effect.parameters.get("factor", 1.0)
                return cv2.convertScaleAbs(frame, alpha=1.0, beta=factor * 50)
            
            return frame
            
        except Exception as e:
            logging.warning(f"Effect {effect.name} failed: {e}")
            return frame


# =============================================================================
# Video Renderer
# =============================================================================

class VideoRenderer:
    """Main video rendering system.
    
    Features:
    - Video composition
    - Audio-video synchronization
    - Post-processing and effects
    - Multiple output formats
    - Quality optimization
    """
    
    def __init__(self):
        """Initialize video renderer."""
        self.logger = logging.getLogger(f"{__name__}.VideoRenderer")
        self.video_processor = VideoProcessor()
        self.logger.info("Video Renderer initialized")
    
    async def render_video(
        self,
        video_frames: np.ndarray,
        audio_path: Optional[str] = None,
        config: Optional[VideoConfig] = None,
        effects: Optional[List[VideoEffect]] = None,
    ) -> str:
        """Render video from frames.
        
        Args:
            video_frames: Array of video frames
            audio_path: Optional path to audio file
            config: Video configuration
            effects: Optional list of video effects
        
        Returns:
            Path to rendered video file
        
        Raises:
            RuntimeError: If rendering fails
        """
        if config is None:
            config = VideoConfig()
        
        if effects is None:
            effects = []
        
        try:
            self.logger.info("Rendering video...")
            
            # Get output dimensions
            width, height = config.get_resolution_tuple()
            
            # Resize frames if needed (process in batches for memory efficiency)
            if video_frames.shape[1:3] != (height, width):
                def resize_batch(batch):
                    return [
                        self.video_processor.resize_frame(frame, (width, height))
                        for frame in batch
                    ]
                
                resized_frames = process_in_batches(
                    list(video_frames),
                    batch_size=32,
                    processor=resize_batch,
                    cleanup_interval=4
                )
                video_frames = np.array(resized_frames)
                clear_memory()
            
            # Apply effects (process in batches for memory efficiency)
            if config.enable_effects and effects:
                fps = config.fps
                frame_list = list(video_frames)
                
                # Track global frame index for time calculation
                global_frame_index = [0]  # Use list to allow modification in nested function
                
                def apply_effects_batch(batch):
                    """Process batch with frame indices for time calculation."""
                    processed = []
                    batch_start = global_frame_index[0]
                    
                    for i, frame in enumerate(batch):
                        current_time = (batch_start + i) / fps
                        processed_frame = frame
                        
                        for effect in effects:
                            if effect.enabled:
                                processed_frame = self.video_processor.apply_effect(
                                    processed_frame, effect, current_time
                                )
                        
                        processed.append(processed_frame)
                    
                    # Update global index for next batch
                    global_frame_index[0] += len(batch)
                    return processed
                
                # Process in batches using helper function
                processed_frames = process_in_batches(
                    frame_list,
                    batch_size=32,
                    processor=apply_effects_batch,
                    cleanup_interval=4
                )
                
                video_frames = np.array(processed_frames)
                clear_memory()
            
            # Save video
            output_dir = Path("./generated_videos")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"video_{uuid.uuid4().hex[:8]}.{config.format.value}"
            
            # Use MoviePy if available for better quality
            if MOVIEPY_AVAILABLE and audio_path:
                self._render_with_moviepy(
                    video_frames, audio_path, str(output_path), config
                )
            else:
                self._render_with_opencv(
                    video_frames, str(output_path), config
                )
            
            self.logger.info(f"Video rendered: {output_path}")
            return str(output_path)
            
        except MemoryError as e:
            self.logger.error("Out of memory during video rendering")
            clear_memory()
            raise RuntimeError(
                "Insufficient memory for video rendering. "
                "Try reducing resolution or number of frames."
            ) from e
        except Exception as e:
            self.logger.error(f"Video rendering failed: {e}", exc_info=True)
            raise RuntimeError(f"Rendering failed: {e}") from e
    
    def _render_with_opencv(
        self,
        frames: np.ndarray,
        output_path: str,
        config: VideoConfig,
    ) -> None:
        """Render video using OpenCV.
        
        Args:
            frames: Video frames
            output_path: Output file path
            config: Video configuration
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            config.fps,
            (width, height),
        )
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
        
        out.release()
    
    def _render_with_moviepy(
        self,
        frames: np.ndarray,
        audio_path: str,
        output_path: str,
        config: VideoConfig,
    ) -> None:
        """Render video with audio using MoviePy.
        
        Args:
            frames: Video frames
            audio_path: Path to audio file
            output_path: Output file path
            config: Video configuration
        """
        try:
            # Create video clip from frames
            clip = mp.ImageSequenceClip(
                [Image.fromarray(frame) for frame in frames],
                fps=config.fps,
            )
            
            # Add audio if available
            if audio_path and Path(audio_path).exists():
                audio_clip = mp.AudioFileClip(audio_path)
                clip = clip.set_audio(audio_clip)
            
            # Write video file
            clip.write_videofile(
                output_path,
                codec=config.codec.value,
                bitrate=f"{config.get_bitrate()}k",
                fps=config.fps,
                audio_codec='aac' if audio_path else None,
            )
            
            clip.close()
            
        except Exception as e:
            self.logger.warning(f"MoviePy rendering failed: {e}")
            # Fallback to OpenCV
            self._render_with_opencv(frames, output_path, config)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "moviepy_available": MOVIEPY_AVAILABLE,
            "audio_available": AUDIO_AVAILABLE,
        }
