"""
Video Processor for Piel Mejorador AI SAM3
==========================================

Frame-by-frame video processing for enhanced skin improvement.
"""

import asyncio
import logging
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes videos frame by frame for skin enhancement.
    
    Features:
    - Frame extraction
    - Frame-by-frame processing
    - Video reconstruction
    - Progress tracking
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize video processor.
        
        Args:
            output_dir: Directory for temporary files
        """
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "piel_mejorador_video"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        frame_interval: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None for all)
            frame_interval: Extract every Nth frame (1 = all frames)
            
        Returns:
            List of frame dictionaries with 'frame_number' and 'image_path'
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Extracting frames from {video_path}: {total_frames} frames at {fps} FPS")
        
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame based on interval
                if frame_count % frame_interval == 0:
                    # Save frame to temporary file
                    frame_path = self.output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    
                    frames.append({
                        "frame_number": frame_count,
                        "image_path": str(frame_path),
                        "timestamp": frame_count / fps if fps > 0 else 0
                    })
                    
                    extracted_count += 1
                    
                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    logger.debug(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    async def process_frames(
        self,
        frames: List[Dict[str, Any]],
        process_func: callable,
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process frames in batches.
        
        Args:
            frames: List of frame dictionaries
            process_func: Async function to process each frame
            batch_size: Number of frames to process in parallel
            
        Returns:
            List of processed frame results
        """
        results = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(frames) + batch_size - 1) // batch_size}")
            
            # Process batch in parallel
            batch_results = await asyncio.gather(*[
                process_func(frame)
                for frame in batch
            ], return_exceptions=True)
            
            # Collect results
            for frame, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing frame {frame['frame_number']}: {result}")
                    results.append({
                        "frame_number": frame["frame_number"],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    results.append({
                        "frame_number": frame["frame_number"],
                        "success": True,
                        "result": result
                    })
        
        return results
    
    async def reconstruct_video(
        self,
        frames: List[Dict[str, Any]],
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v"
    ) -> str:
        """
        Reconstruct video from processed frames.
        
        Args:
            frames: List of frame dictionaries with processed image paths
            output_path: Path for output video
            fps: Frames per second for output video
            codec: Video codec to use
            
        Returns:
            Path to reconstructed video
        """
        if not frames:
            raise ValueError("No frames provided for video reconstruction")
        
        # Get frame dimensions from first frame
        first_frame_path = frames[0].get("processed_path") or frames[0].get("image_path")
        first_frame = cv2.imread(first_frame_path)
        
        if first_frame is None:
            raise ValueError(f"Cannot read first frame: {first_frame_path}")
        
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Reconstructing video: {len(frames)} frames at {fps} FPS")
        
        try:
            for frame_info in sorted(frames, key=lambda x: x["frame_number"]):
                frame_path = frame_info.get("processed_path") or frame_info.get("image_path")
                
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Cannot read frame {frame_info['frame_number']}: {frame_path}")
                    continue
                
                # Resize if necessary
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
        
        finally:
            out.release()
        
        logger.info(f"Video reconstructed: {output_path}")
        return output_path
    
    def cleanup_temp_files(self, frames: List[Dict[str, Any]]):
        """Clean up temporary frame files."""
        for frame in frames:
            frame_path = Path(frame.get("image_path", ""))
            if frame_path.exists():
                try:
                    frame_path.unlink()
                    logger.debug(f"Cleaned up frame: {frame_path}")
                except Exception as e:
                    logger.warning(f"Error cleaning up frame {frame_path}: {e}")




