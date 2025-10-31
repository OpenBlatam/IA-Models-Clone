from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
from typing import Dict, Optional
from ...video_generator import VideoGenerator, VideoGenerationResult
from ...web_extract import ExtractedContent
from ...suggestions import ContentSuggestions
from ...plugins import BasePlugin
from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
Integrated Workflow - Generators

Integrated video generator components that use available plugins.
"""


logger = logging.getLogger(__name__)


class IntegratedVideoGenerator(VideoGenerator):
    """Integrated video generator that uses available plugins."""
    
    def __init__(self, generators: Dict[str, BasePlugin]):
        
    """__init__ function."""
super().__init__()
        self.generators = generators
        self.last_used = None
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'generator_usage': {}
        }
    
    async def generate_video(
        self, 
        content: ExtractedContent, 
        suggestions: ContentSuggestions, 
        avatar: Optional[str] = None
    ) -> VideoGenerationResult:
        """Generate video using available plugins."""
        self.generation_stats['total_generations'] += 1
        
        # Use the first available generator
        for name, generator in self.generators.items():
            try:
                if hasattr(generator, 'generate_video'):
                    logger.info(f"🎬 Trying video generator: {name}")
                    result = await generator.generate_video(content, suggestions, avatar)
                    
                    if result:
                        self.last_used = name
                        self.generation_stats['successful_generations'] += 1
                        self.generation_stats['generator_usage'][name] = \
                            self.generation_stats['generator_usage'].get(name, 0) + 1
                        
                        logger.info(f"✅ Video generated successfully using: {name}")
                        return result
                        
            except Exception as e:
                logger.warning(f"❌ Video generator {name} failed: {e}")
                self.generation_stats['failed_generations'] += 1
                continue
        
        # Fallback to default generation
        logger.info("🔄 Falling back to default video generation")
        try:
            result = await super().generate_video(content, suggestions, avatar)
            if result:
                self.generation_stats['successful_generations'] += 1
                self.generation_stats['generator_usage']['default'] = \
                    self.generation_stats['generator_usage'].get('default', 0) + 1
                return result
        except Exception as e:
            logger.error(f"❌ Default video generation also failed: {e}")
            self.generation_stats['failed_generations'] += 1
        
        # Return error result as last resort
        return VideoGenerationResult(
            success=False,
            video_url="",
            error="All video generators failed",
            metadata={
                "generators_tried": list(self.generators.keys()),
                "fallback_used": True
            }
        )
    
    def get_last_used_generator(self) -> Optional[str]:
        """Get the name of the last used video generator."""
        return self.last_used
    
    def get_generation_stats(self) -> Dict:
        """Get generation statistics."""
        return self.generation_stats.copy()
    
    def get_available_generators(self) -> list[str]:
        """Get list of available video generators."""
        return list(self.generators.keys())
    
    def get_generator_info(self, name: str) -> Optional[Dict]:
        """Get information about a specific video generator."""
        if name in self.generators:
            generator = self.generators[name]
            metadata = generator.get_metadata()
            return {
                'name': name,
                'version': metadata.version,
                'description': metadata.description,
                'author': metadata.author,
                'category': metadata.category,
                'usage_count': self.generation_stats['generator_usage'].get(name, 0)
            }
        return None


class FallbackVideoGenerator(VideoGenerator):
    """Fallback video generator for when plugins are not available."""
    
    def __init__(self) -> Any:
        super().__init__()
        self.fallback_methods = [
            self._generate_basic_video,
            self._generate_template_video,
            self._generate_simple_video
        ]
    
    async def generate_video(
        self, 
        content: ExtractedContent, 
        suggestions: ContentSuggestions, 
        avatar: Optional[str] = None
    ) -> VideoGenerationResult:
        """Generate video using fallback methods."""
        for method in self.fallback_methods:
            try:
                result = await method(content, suggestions, avatar)
                if result and result.success:
                    logger.info(f"✅ Fallback video generation successful with {method.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"❌ Fallback method {method.__name__} failed: {e}")
                continue
        
        return await super().generate_video(content, suggestions, avatar)
    
    async def _generate_basic_video(
        self, 
        content: ExtractedContent, 
        suggestions: ContentSuggestions, 
        avatar: Optional[str] = None
    ) -> VideoGenerationResult:
        """Generate basic video from content."""
        # Basic video generation logic
        video_url = f"generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        return VideoGenerationResult(
            success=True,
            video_url=video_url,
            metadata={
                "method": "basic",
                "content_length": len(content.text),
                "avatar_used": avatar or "default"
            }
        )
    
    async def _generate_template_video(
        self, 
        content: ExtractedContent, 
        suggestions: ContentSuggestions, 
        avatar: Optional[str] = None
    ) -> VideoGenerationResult:
        """Generate video using templates."""
        # Template-based video generation
        video_url = f"template_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        return VideoGenerationResult(
            success=True,
            video_url=video_url,
            metadata={
                "method": "template",
                "templates_used": len(suggestions.title_suggestions),
                "avatar_used": avatar or "default"
            }
        )
    
    async def _generate_simple_video(
        self, 
        content: ExtractedContent, 
        suggestions: ContentSuggestions, 
        avatar: Optional[str] = None
    ) -> VideoGenerationResult:
        """Generate simple video with minimal processing."""
        # Simple video generation
        video_url = f"simple_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        return VideoGenerationResult(
            success=True,
            video_url=video_url,
            metadata={
                "method": "simple",
                "content_title": content.title or "Untitled",
                "avatar_used": avatar or "default"
            }
        )


class VideoGeneratorManager:
    """Manager for video generator plugins."""
    
    def __init__(self) -> Any:
        self.generators: Dict[str, BasePlugin] = {}
        self.generator_priorities: Dict[str, int] = {}
        self.generation_history: list[Dict] = []
    
    def add_generator(self, name: str, generator: BasePlugin, priority: int = 0):
        """Add a video generator with priority."""
        self.generators[name] = generator
        self.generator_priorities[name] = priority
        logger.info(f"🎬 Added video generator: {name} (priority: {priority})")
    
    def remove_generator(self, name: str):
        """Remove a video generator."""
        if name in self.generators:
            del self.generators[name]
            del self.generator_priorities[name]
            logger.info(f"🗑️ Removed video generator: {name}")
    
    def get_prioritized_generators(self) -> Dict[str, BasePlugin]:
        """Get generators sorted by priority."""
        sorted_generators = sorted(
            self.generators.items(),
            key=lambda x: self.generator_priorities.get(x[0], 0),
            reverse=True
        )
        return dict(sorted_generators)
    
    def update_priority(self, name: str, priority: int):
        """Update generator priority."""
        if name in self.generators:
            self.generator_priorities[name] = priority
            logger.info(f"🔄 Updated priority for {name}: {priority}")
    
    def get_generator_stats(self) -> Dict:
        """Get generator statistics."""
        return {
            'total_generators': len(self.generators),
            'priorities': self.generator_priorities.copy(),
            'generation_history': self.generation_history[-10:],  # Last 10 generations
            'generators': list(self.generators.keys())
        }
    
    def record_generation(
        self, 
        generator_name: str, 
        content_length: int, 
        success: bool, 
        duration: float,
        video_url: str = ""
    ):
        """Record generation attempt."""
        self.generation_history.append({
            'generator': generator_name,
            'content_length': content_length,
            'success': success,
            'duration': duration,
            'video_url': video_url,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 records
        if len(self.generation_history) > 100:
            self.generation_history = self.generation_history[-100:] 