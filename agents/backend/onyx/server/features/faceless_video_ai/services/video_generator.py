"""
Video Generator Service
Generates AI images and creates video sequences
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from functools import lru_cache

from .ai_providers.image_providers import get_image_provider, ImageProvider

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Generates video sequences from AI-generated images"""

    def __init__(self, output_dir: Optional[str] = None, image_provider: Optional[ImageProvider] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache = {}
        self.image_provider = image_provider or get_image_provider()
        self.max_retries = 3
        self.retry_delay = 2.0

    async def generate_images_for_segments(
        self,
        segments: List[Dict[str, Any]],
        style: str = "realistic",
        resolution: str = "1920x1080"
    ) -> List[Dict[str, Any]]:
        """
        Generate AI images for each script segment
        
        Args:
            segments: List of script segments
            style: Video style (realistic, animated, abstract, etc.)
            resolution: Image resolution (e.g., "1920x1080")
            
        Returns:
            List of segment dicts with image paths
        """
        width, height = map(int, resolution.split('x'))
        
        logger.info(f"Generating {len(segments)} images with style '{style}'")
        
        tasks = []
        for i, segment in enumerate(segments):
            task = self._generate_single_image(
                segment=segment,
                index=i,
                style=style,
                width=width,
                height=height
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Update segments with image paths
        for segment, image_data in zip(segments, results):
            segment["image_path"] = image_data["path"]
            segment["image_url"] = image_data.get("url")
        
        logger.info(f"Successfully generated {len(results)} images")
        return segments

    async def _generate_single_image(
        self,
        segment: Dict[str, Any],
        index: int,
        style: str,
        width: int,
        height: int
    ) -> Dict[str, Any]:
        """Generate a single AI image for a segment with retry logic"""
        keywords = segment.get("keywords", [])
        text = segment.get("text", "")
        
        # Create prompt for image generation
        prompt = self._create_image_prompt(text, keywords, style)
        
        # Check cache first
        cache_key = f"{style}_{hash(prompt)}_{width}x{height}"
        if cache_key in self.image_cache:
            logger.debug(f"Using cached image for segment {index}")
            return self.image_cache[cache_key]
        
        # Generate image with retry logic
        image_path = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                image_path = await self._generate_ai_image(
                    prompt=prompt,
                    style=style,
                    width=width,
                    height=height,
                    index=index
                )
                break  # Success
            except Exception as e:
                last_error = e
                logger.warning(f"Image generation attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        if image_path is None:
            logger.error(f"Failed to generate image after {self.max_retries} attempts")
            raise RuntimeError(f"Image generation failed: {str(last_error)}")
        
        result = {
            "path": str(image_path),
            "prompt": prompt,
            "style": style,
        }
        
        # Cache result
        self.image_cache[cache_key] = result
        
        return result

    def _create_image_prompt(
        self,
        text: str,
        keywords: List[str],
        style: str
    ) -> str:
        """Create prompt for AI image generation"""
        # Extract main concepts from text
        main_keywords = " ".join(keywords[:3]) if keywords else "abstract concept"
        
        style_prompts = {
            "realistic": f"high quality realistic image, professional photography, {main_keywords}, detailed, sharp focus",
            "animated": f"beautiful animated illustration, vibrant colors, {main_keywords}, cartoon style, modern animation",
            "abstract": f"abstract art, creative composition, {main_keywords}, artistic, colorful, modern design",
            "minimalist": f"minimalist design, clean composition, {main_keywords}, simple, elegant, modern",
            "dynamic": f"dynamic composition, energetic, {main_keywords}, motion, vibrant, engaging",
        }
        
        base_prompt = style_prompts.get(style, style_prompts["realistic"])
        
        # Add faceless video specific instructions
        prompt = f"{base_prompt}, no people, no faces, concept visualization, suitable for faceless video"
        
        return prompt

    async def _generate_ai_image(
        self,
        prompt: str,
        style: str,
        width: int,
        height: int,
        index: int
    ) -> Path:
        """
        Generate AI image using configured image provider
        """
        try:
            # Use the configured image provider
            image_path = await self.image_provider.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                style=style
            )
            
            # Verify image exists and is valid
            if not image_path.exists():
                raise FileNotFoundError(f"Generated image not found: {image_path}")
            
            # Verify it's a valid image
            try:
                img = Image.open(image_path)
                img.verify()
            except Exception as e:
                raise ValueError(f"Invalid image file: {str(e)}")
            
            logger.debug(f"Generated image: {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise

    async def create_image_sequence(
        self,
        segments: List[Dict[str, Any]],
        fps: int = 30,
        image_duration: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Create image sequence with timing for video composition
        
        Args:
            segments: Segments with image paths
            fps: Frames per second
            image_duration: Duration per image in seconds
            
        Returns:
            List of frame data for video composition
        """
        frames = []
        
        for segment in segments:
            image_path = segment.get("image_path")
            if not image_path or not Path(image_path).exists():
                logger.warning(f"Image not found for segment {segment.get('index')}")
                continue
            
            duration = segment.get("duration", image_duration)
            num_frames = int(duration * fps)
            
            frames.append({
                "image_path": image_path,
                "duration": duration,
                "num_frames": num_frames,
                "segment_index": segment.get("index"),
            })
        
        logger.info(f"Created image sequence with {len(frames)} frames")
        return frames

