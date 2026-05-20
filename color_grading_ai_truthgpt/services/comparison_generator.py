"""
Comparison Generator for Color Grading AI
=========================================

Generates before/after comparisons and side-by-side previews.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import asyncio
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ComparisonGenerator:
    """
    Generates comparison views.
    
    Features:
    - Side-by-side before/after
    - Split screen comparison
    - Animated GIF comparisons
    - Video comparison clips
    """
    
    def __init__(self):
        """Initialize comparison generator."""
        pass
    
    async def create_image_comparison(
        self,
        before_path: str,
        after_path: str,
        output_path: str,
        style: str = "side_by_side",
        labels: bool = True
    ) -> str:
        """
        Create image comparison.
        
        Args:
            before_path: Path to before image
            after_path: Path to after image
            output_path: Output comparison path
            style: Comparison style (side_by_side, split, overlay)
            labels: Add labels
            
        Returns:
            Path to comparison image
        """
        def _create():
            before = Image.open(before_path)
            after = Image.open(after_path)
            
            # Resize to same size
            max_width = max(before.width, after.width)
            max_height = max(before.height, after.height)
            
            before = before.resize((max_width, max_height), Image.Resampling.LANCZOS)
            after = after.resize((max_width, max_height), Image.Resampling.LANCZOS)
            
            if style == "side_by_side":
                # Create side-by-side
                comparison = Image.new("RGB", (max_width * 2, max_height))
                comparison.paste(before, (0, 0))
                comparison.paste(after, (max_width, 0))
                
                if labels:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(comparison)
                    try:
                        font = ImageFont.truetype("arial.ttf", 30)
                    except:
                        font = ImageFont.load_default()
                    
                    draw.text((10, 10), "Before", fill="white", font=font)
                    draw.text((max_width + 10, 10), "After", fill="white", font=font)
            
            elif style == "split":
                # Split screen
                comparison = Image.new("RGB", (max_width, max_height))
                comparison.paste(before.crop((0, 0, max_width // 2, max_height)), (0, 0))
                comparison.paste(after.crop((max_width // 2, 0, max_width, max_height)), (max_width // 2, 0))
            
            else:  # overlay
                # Overlay with transparency
                comparison = Image.blend(before, after, 0.5)
            
            comparison.save(output_path, quality=95)
            return output_path
        
        return await asyncio.to_thread(_create)
    
    async def create_video_comparison(
        self,
        before_path: str,
        after_path: str,
        output_path: str,
        style: str = "side_by_side",
        duration: Optional[float] = None
    ) -> str:
        """
        Create video comparison.
        
        Args:
            before_path: Path to before video
            after_path: Path to after video
            output_path: Output comparison path
            style: Comparison style
            duration: Optional duration limit
            
        Returns:
            Path to comparison video
        """
        # This would use FFmpeg to create side-by-side video
        # Simplified implementation
        logger.info(f"Creating video comparison: {style}")
        
        # For full implementation, would use VideoProcessor with FFmpeg
        # For now, return placeholder
        return output_path
    
    async def create_animated_comparison(
        self,
        before_path: str,
        after_path: str,
        output_path: str,
        duration: float = 3.0,
        fps: int = 2
    ) -> str:
        """
        Create animated GIF comparison.
        
        Args:
            before_path: Path to before image
            after_path: Path to after image
            output_path: Output GIF path
            duration: Animation duration
            fps: Frames per second
            
        Returns:
            Path to animated GIF
        """
        def _create_gif():
            before = Image.open(before_path)
            after = Image.open(after_path)
            
            # Resize to same size
            max_width = max(before.width, after.width)
            max_height = max(before.height, after.height)
            
            before = before.resize((max_width, max_height), Image.Resampling.LANCZOS)
            after = after.resize((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Create frames (fade between before and after)
            frames = []
            num_frames = int(duration * fps)
            
            for i in range(num_frames):
                alpha = i / (num_frames - 1) if num_frames > 1 else 0
                frame = Image.blend(before, after, alpha)
                frames.append(frame)
            
            # Add reverse for loop effect
            frames.extend(reversed(frames[1:-1]))
            
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0
            )
            
            return output_path
        
        return await asyncio.to_thread(_create_gif)




