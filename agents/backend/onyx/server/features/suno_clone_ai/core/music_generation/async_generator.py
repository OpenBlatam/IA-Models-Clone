"""
Async Generator Module

Asynchronous music generation for better performance.
"""

from typing import Optional, List, Dict, Any
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


class AsyncMusicGenerator:
    """
    Asynchronous wrapper for music generators.
    """
    
    def __init__(
        self,
        generator_type: str = "audiocraft",
        model_name: Optional[str] = None,
        max_workers: int = 2
    ):
        self.generator_type = generator_type
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._generator = None
    
    def _get_generator(self):
        """Get or create generator."""
        if self._generator is None:
            from .generators import create_generator
            self._generator = create_generator(
                generator_type=self.generator_type,
                model_name=self.model_name
            )
        return self._generator
    
    async def generate_async(
        self,
        prompt: str,
        duration: int = 30,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music asynchronously.
        
        Args:
            prompt: Text prompt
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            Generated audio
        """
        loop = asyncio.get_event_loop()
        generator = self._get_generator()
        
        return await loop.run_in_executor(
            self.executor,
            generator.generate,
            prompt,
            duration,
            **kwargs
        )
    
    async def generate_batch_async(
        self,
        prompts: List[str],
        duration: int = 30,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Generate multiple tracks asynchronously.
        
        Args:
            prompts: List of prompts
            duration: Duration for each
            **kwargs: Additional parameters
            
        Returns:
            List of generated audio
        """
        tasks = [
            self.generate_async(prompt, duration, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
    
    async def generate_with_post_processing_async(
        self,
        prompt: str,
        duration: int = 30,
        **kwargs
    ) -> np.ndarray:
        """
        Generate and post-process asynchronously.
        
        Args:
            prompt: Text prompt
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            Processed audio
        """
        # Generate
        audio = await self.generate_async(prompt, duration, **kwargs)
        
        # Post-process in executor
        loop = asyncio.get_event_loop()
        from .post_processing import AudioPostProcessor
        
        processor = AudioPostProcessor()
        processed = await loop.run_in_executor(
            self.executor,
            processor.process_full_pipeline,
            audio
        )
        
        return processed
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self._generator:
            del self._generator
            self._generator = None


class AsyncPipeline:
    """
    Asynchronous pipeline for music generation.
    """
    
    def __init__(
        self,
        generator_type: str = "audiocraft",
        max_workers: int = 2
    ):
        self.generator_type = generator_type
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pipeline = None
    
    def _get_pipeline(self):
        """Get or create pipeline."""
        if self._pipeline is None:
            from .pipeline import MusicGenerationPipeline
            self._pipeline = MusicGenerationPipeline(
                generator_type=self.generator_type
            )
        return self._pipeline
    
    async def generate_async(
        self,
        prompt: str,
        duration: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run pipeline asynchronously.
        
        Args:
            prompt: Text prompt
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            Generation result
        """
        loop = asyncio.get_event_loop()
        pipeline = self._get_pipeline()
        
        return await loop.run_in_executor(
            self.executor,
            pipeline.generate,
            prompt,
            duration,
            **kwargs
        )
    
    async def generate_batch_async(
        self,
        prompts: List[str],
        duration: int = 30,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate batch asynchronously.
        
        Args:
            prompts: List of prompts
            duration: Duration for each
            **kwargs: Additional parameters
            
        Returns:
            List of results
        """
        tasks = [
            self.generate_async(prompt, duration, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)















