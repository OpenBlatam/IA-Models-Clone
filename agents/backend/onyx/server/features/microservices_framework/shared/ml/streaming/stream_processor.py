"""
Stream Processor
Streaming utilities for real-time processing.
"""

import asyncio
from typing import AsyncIterator, Iterator, Callable, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Process data streams asynchronously."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_stream(
        self,
        stream: AsyncIterator[Any],
        processor: Callable[[Any], Any],
    ) -> AsyncIterator[Any]:
        """Process stream with async processor."""
        async for item in stream:
            async with self.semaphore:
                result = await processor(item)
                yield result
    
    async def process_batch_stream(
        self,
        stream: AsyncIterator[Any],
        processor: Callable[[List[Any]], List[Any]],
    ) -> AsyncIterator[Any]:
        """Process stream in batches."""
        batch = []
        
        async for item in stream:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                async with self.semaphore:
                    results = await processor(batch)
                    for result in results:
                        yield result
                    batch = []
        
        # Process remaining items
        if batch:
            async with self.semaphore:
                results = await processor(batch)
                for result in results:
                    yield result


class TokenStreamer:
    """Stream tokens from language models."""
    
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
    
    async def stream_tokens(
        self,
        prompt: str,
        max_length: int = 100,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generated tokens."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate tokens
        for token_id in self.model.generate_stream(inputs, max_length=max_length, **kwargs):
            token = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield token
    
    async def stream_text(
        self,
        prompt: str,
        max_length: int = 100,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream complete text."""
        accumulated = ""
        async for token in self.stream_tokens(prompt, max_length, **kwargs):
            accumulated += token
            yield accumulated


class ImageStreamer:
    """Stream image generation progress."""
    
    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
    
    async def stream_generation(
        self,
        prompt: str,
        num_steps: int = 50,
        **kwargs
    ) -> AsyncIterator[Any]:
        """Stream intermediate images during generation."""
        for step in range(num_steps):
            # Get intermediate result
            intermediate = await self.pipeline.generate_step(
                prompt,
                step=step,
                total_steps=num_steps,
                **kwargs
            )
            yield intermediate



