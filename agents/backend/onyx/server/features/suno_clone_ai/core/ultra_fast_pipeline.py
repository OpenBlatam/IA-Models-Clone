"""
Ultra-Fast Music Generation Pipeline

Optimized pipeline with:
- Parallel processing
- Streaming generation
- Memory-efficient batch processing
- Advanced caching
- Smart resource management
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, AsyncIterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import numpy as np
from dataclasses import dataclass

from .ultra_fast_generator import UltraFastMusicGenerator, get_ultra_fast_generator

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for generation."""
    duration: int = 30
    temperature: float = 1.0
    guidance_scale: float = 3.0
    top_k: int = 250
    top_p: float = 0.0
    batch_size: int = 4
    use_cache: bool = True


class UltraFastPipeline:
    """
    Ultra-fast pipeline with advanced optimizations.
    """
    
    def __init__(
        self,
        generator: Optional[UltraFastMusicGenerator] = None,
        max_workers: int = 4,
        use_process_pool: bool = False
    ):
        """
        Initialize ultra-fast pipeline.
        
        Args:
            generator: Ultra-fast generator instance
            max_workers: Max workers for parallel processing
            use_process_pool: Use process pool instead of thread pool
        """
        self.generator = generator or get_ultra_fast_generator()
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        
        # Setup executor
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def generate_streaming(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        chunk_size: int = 5
    ) -> AsyncIterator[np.ndarray]:
        """
        Stream generation in chunks for real-time playback.
        
        Args:
            prompt: Text prompt
            config: Generation configuration
            chunk_size: Size of each chunk in seconds
            
        Yields:
            Audio chunks
        """
        config = config or GenerationConfig()
        
        # Generate full audio
        full_audio = await self.generator.generate_async(
            text=prompt,
            duration=config.duration,
            temperature=config.temperature,
            guidance_scale=config.guidance_scale,
            top_k=config.top_k,
            top_p=config.top_p
        )
        
        # Stream in chunks
        sample_rate = 32000  # Default sample rate
        chunk_samples = chunk_size * sample_rate
        
        for i in range(0, len(full_audio), chunk_samples):
            chunk = full_audio[i:i + chunk_samples]
            yield chunk
            await asyncio.sleep(0.01)  # Small delay for streaming
    
    async def generate_parallel(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[np.ndarray]:
        """
        Generate multiple tracks in parallel.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of audio arrays
        """
        config = config or GenerationConfig()
        
        # Create tasks
        tasks = [
            self.generator.generate_async(
                text=prompt,
                duration=config.duration,
                temperature=config.temperature,
                guidance_scale=config.guidance_scale,
                top_k=config.top_k,
                top_p=config.top_p
            )
            for prompt in prompts
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        audio_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error generating for prompt {i}: {result}")
                audio_list.append(None)
            else:
                audio_list.append(result)
        
        return audio_list
    
    async def generate_with_priority(
        self,
        prompts: List[tuple[str, int]],  # (prompt, priority)
        config: Optional[GenerationConfig] = None
    ) -> List[np.ndarray]:
        """
        Generate with priority queue.
        
        Args:
            prompts: List of (prompt, priority) tuples
            config: Generation configuration
            
        Returns:
            List of audio arrays in priority order
        """
        config = config or GenerationConfig()
        
        # Sort by priority
        sorted_prompts = sorted(prompts, key=lambda x: x[1], reverse=True)
        
        # Generate in priority order
        results = []
        for prompt, _ in sorted_prompts:
            audio = await self.generator.generate_async(
                text=prompt,
                duration=config.duration,
                temperature=config.temperature,
                guidance_scale=config.guidance_scale
            )
            results.append(audio)
        
        return results
    
    def generate_batch_optimized(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[np.ndarray]:
        """
        Optimized batch generation with smart batching.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of audio arrays
        """
        config = config or GenerationConfig()
        
        # Use generator's batch method
        return self.generator.generate_batch(
            texts=prompts,
            duration=config.duration,
            batch_size=config.batch_size
        )
    
    async def generate_with_retry(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Generate with automatic retry on failure.
        
        Args:
            prompt: Text prompt
            config: Generation configuration
            max_retries: Maximum retry attempts
            
        Returns:
            Generated audio
        """
        config = config or GenerationConfig()
        
        for attempt in range(max_retries):
            try:
                audio = await self.generator.generate_async(
                    text=prompt,
                    duration=config.duration,
                    temperature=config.temperature,
                    guidance_scale=config.guidance_scale
                )
                return audio
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError("Max retries exceeded")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=False)


class StreamingGenerator:
    """
    Real-time streaming generation.
    """
    
    def __init__(self, generator: Optional[UltraFastMusicGenerator] = None):
        """Initialize streaming generator."""
        self.generator = generator or get_ultra_fast_generator()
        self._buffer = []
    
    async def generate_chunk(
        self,
        prompt: str,
        chunk_duration: int = 5,
        total_duration: int = 30
    ) -> np.ndarray:
        """
        Generate a single chunk.
        
        Args:
            prompt: Text prompt
            chunk_duration: Duration of chunk in seconds
            total_duration: Total desired duration
            
        Returns:
            Audio chunk
        """
        # For now, generate full and split
        # In production, would use incremental generation
        full_audio = await self.generator.generate_async(
            text=prompt,
            duration=total_duration
        )
        
        sample_rate = 32000
        chunk_samples = chunk_duration * sample_rate
        
        if len(self._buffer) == 0:
            self._buffer = [full_audio]
        
        if len(self._buffer[0]) >= chunk_samples:
            chunk = self._buffer[0][:chunk_samples]
            self._buffer[0] = self._buffer[0][chunk_samples:]
            return chunk
        
        return np.array([])
    
    def clear_buffer(self) -> None:
        """Clear generation buffer."""
        self._buffer.clear()


class MemoryEfficientBatchProcessor:
    """
    Memory-efficient batch processing with smart memory management.
    """
    
    def __init__(
        self,
        generator: Optional[UltraFastMusicGenerator] = None,
        max_memory_gb: float = 8.0
    ):
        """
        Initialize batch processor.
        
        Args:
            generator: Generator instance
            max_memory_gb: Maximum memory to use in GB
        """
        self.generator = generator or get_ultra_fast_generator()
        self.max_memory_gb = max_memory_gb
    
    def calculate_optimal_batch_size(
        self,
        duration: int,
        sample_rate: int = 32000
    ) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return 1
        
        # Estimate memory per sample
        samples = duration * sample_rate
        bytes_per_sample = 4  # float32
        memory_per_sample_gb = (samples * bytes_per_sample) / (1024**3)
        
        # Get available memory
        free_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated()
        ) / (1024**3)
        
        # Calculate batch size (leave 2GB for model and overhead)
        usable_memory = min(free_memory_gb, self.max_memory_gb) - 2.0
        
        if usable_memory < memory_per_sample_gb:
            return 1
        
        batch_size = int(usable_memory / memory_per_sample_gb)
        return max(1, min(batch_size, 8))  # Cap at 8
    
    async def process_batch_smart(
        self,
        prompts: List[str],
        duration: int = 30
    ) -> List[np.ndarray]:
        """
        Process batch with smart memory management.
        
        Args:
            prompts: List of prompts
            duration: Duration in seconds
            
        Returns:
            List of audio arrays
        """
        # Calculate optimal batch size
        batch_size = self.calculate_optimal_batch_size(duration)
        
        logger.info(f"Using batch size: {batch_size} for {len(prompts)} prompts")
        
        # Process in optimal batches
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Generate batch
            batch_results = self.generator.generate_batch(
                texts=batch_prompts,
                duration=duration,
                batch_size=batch_size
            )
            
            results.extend(batch_results)
            
            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results













