from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable
from asyncio import Semaphore, Queue
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import hashlib
import aioredis
import orjson
from cachetools import TTLCache
from ...shared.logging import get_logger
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            import textstat
            from keybert import KeyBERT
            import spacy
            from transformers import pipeline
from typing import Any, List, Dict, Optional
import logging
"""
Async NLP Processor for LinkedIn Posts
=====================================

Ultra-fast async NLP processing with optimized patterns,
connection pooling, and batch operations for maximum speed.
"""




logger = get_logger(__name__)


class AsyncNLPProcessor:
    """
    Ultra-fast async NLP processor with advanced optimizations.
    
    Features:
    - Async/await patterns for non-blocking operations
    - Connection pooling for Redis
    - Batch processing capabilities
    - Intelligent caching with TTL
    - Parallel task execution
    - Performance monitoring
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize async NLP processor."""
        self.redis_url = redis_url
        self.redis_client = None
        self.redis_pool = None
        
        # Memory cache for ultra-fast access
        self.memory_cache = TTLCache(maxsize=2000, ttl=1800)  # 30 minutes
        
        # Thread pool for CPU-intensive NLP tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Semaphore for concurrency control
        self.processing_semaphore = Semaphore(20)
        
        # Batch processing queue
        self.batch_queue = Queue()
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_operations": 0,
            "average_processing_time": 0.0,
            "concurrent_operations": 0,
        }
        
        # Initialize Redis connection
        asyncio.create_task(self._initialize_redis())
        asyncio.create_task(self._process_batch_queue())
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection pool."""
        try:
            self.redis_pool = aioredis.from_url(
                self.redis_url,
                max_connections=30,
                decode_responses=False,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.redis_client = self.redis_pool
            logger.info("Async NLP Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis for async NLP: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, text: str, operation: str = "enhance") -> str:
        """Generate cache key for text and operation."""
        content = f"{operation}:{text}"
        return f"async_nlp:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache with async operations."""
        try:
            # Check memory cache first (fastest)
            if cache_key in self.memory_cache:
                self.metrics["cache_hits"] += 1
                return self.memory_cache[cache_key]
            
            # Check Redis cache
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    result = orjson.loads(cached_data)
                    self.memory_cache[cache_key] = result
                    self.metrics["cache_hits"] += 1
                    return result
            
            self.metrics["cache_misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Async cache get error: {e}")
            return None
    
    async def _set_cache(self, cache_key: str, result: Dict[str, Any], ttl: int = 1800):
        """Set result in cache with async operations."""
        try:
            # Set in memory cache immediately
            self.memory_cache[cache_key] = result
            
            # Set in Redis cache asynchronously
            if self.redis_client:
                serialized = orjson.dumps(result)
                await self.redis_client.setex(cache_key, ttl, serialized)
                
        except Exception as e:
            logger.error(f"Async cache set error: {e}")
    
    async def _process_nlp_tasks_async(self, text: str) -> Dict[str, Any]:
        """Process NLP tasks asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Create async tasks for parallel processing
        tasks = [
            loop.run_in_executor(self.thread_pool, self._analyze_sentiment_async, text),
            loop.run_in_executor(self.thread_pool, self._analyze_readability_async, text),
            loop.run_in_executor(self.thread_pool, self._extract_keywords_async, text),
            loop.run_in_executor(self.thread_pool, self._detect_entities_async, text),
            loop.run_in_executor(self.thread_pool, self._improve_text_async, text),
        ]
        
        # Execute all tasks concurrently
        sentiment, readability, keywords, entities, improved = await asyncio.gather(*tasks)
        
        return {
            "sentiment": sentiment,
            "readability": readability,
            "keywords": keywords,
            "entities": entities,
            "improved_text": improved,
        }
    
    def _analyze_sentiment_async(self, text: str) -> Dict[str, float]:
        """Analyze sentiment asynchronously."""
        try:
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(text)
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    
    def _analyze_readability_async(self, text: str) -> Dict[str, float]:
        """Analyze readability asynchronously."""
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
            }
        except Exception as e:
            logger.error(f"Readability analysis error: {e}")
            return {"flesch_reading_ease": 0.0}
    
    def _extract_keywords_async(self, text: str) -> List[str]:
        """Extract keywords asynchronously."""
        try:
            kw_model = KeyBERT('all-MiniLM-L6-v2')
            keywords = kw_model.extract_keywords(text, top_n=5)
            return [kw[0] for kw in keywords]
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []
    
    def _detect_entities_async(self, text: str) -> List[tuple]:
        """Detect entities asynchronously."""
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            logger.error(f"Entity detection error: {e}")
            return []
    
    def _improve_text_async(self, text: str) -> str:
        """Improve text asynchronously."""
        try:
            rewriter = pipeline("text2text-generation", model="google/flan-t5-base")
            result = rewriter(f"Improve this LinkedIn post: {text}", max_length=256)
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Text improvement error: {e}")
            return text
    
    async def enhance_post_async(self, text: str) -> Dict[str, Any]:
        """
        Ultra-fast async post enhancement with caching and parallel processing.
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        async with self.processing_semaphore:
            try:
                # Generate cache key
                cache_key = self._generate_cache_key(text)
                
                # Try to get from cache
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    processing_time = time.time() - start_time
                    self._update_metrics(processing_time)
                    return {
                        "original": text,
                        "enhanced": cached_result,
                        "processing_time": processing_time,
                        "cached": True,
                        "async_optimized": True,
                    }
                
                # Process NLP tasks asynchronously
                nlp_results = await self._process_nlp_tasks_async(text)
                
                # Compile result
                result = {
                    "original": text,
                    "improved_text": nlp_results["improved_text"],
                    "sentiment": nlp_results["sentiment"],
                    "readability": nlp_results["readability"],
                    "keywords": nlp_results["keywords"],
                    "entities": nlp_results["entities"],
                    "grammar_issues": 0,  # Simplified for speed
                    "grammar_suggestions": [],  # Simplified for speed
                }
                
                # Cache result asynchronously
                asyncio.create_task(self._set_cache(cache_key, result))
                
                # Update metrics
                processing_time = time.time() - start_time
                self._update_metrics(processing_time)
                
                return {
                    "original": text,
                    "enhanced": result,
                    "processing_time": processing_time,
                    "cached": False,
                    "async_optimized": True,
                }
                
            except Exception as e:
                logger.error(f"Async NLP enhancement error: {e}")
                processing_time = time.time() - start_time
                self._update_metrics(processing_time)
                
                return {
                    "original": text,
                    "enhanced": {"error": str(e)},
                    "processing_time": processing_time,
                    "cached": False,
                    "async_optimized": True,
                }
    
    async def enhance_multiple_posts_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Enhance multiple posts with async batch processing."""
        # Process posts concurrently with semaphore control
        tasks = []
        for text in texts:
            task = asyncio.create_task(self.enhance_post_async(text))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "original": texts[i],
                    "enhanced": {"error": str(result)},
                    "processing_time": 0,
                    "cached": False,
                    "async_optimized": True,
                })
            else:
                processed_results.append(result)
        
        self.metrics["batch_operations"] += 1
        return processed_results
    
    async def _process_batch_queue(self) -> Any:
        """Process batch queue for background operations."""
        while True:
            try:
                # Get batch from queue
                batch = await asyncio.wait_for(self.batch_queue.get(), timeout=1.0)
                
                # Process batch
                if batch:
                    await self.enhance_multiple_posts_async(batch)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics."""
        current_avg = self.metrics["average_processing_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get async performance metrics."""
        total_requests = self.metrics["total_requests"]
        cache_hit_rate = (
            self.metrics["cache_hits"] / total_requests * 100 
            if total_requests > 0 else 0
        )
        
        return {
            **self.metrics,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "total_requests": total_requests,
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
            "concurrent_operations": self.processing_semaphore._value,
        }
    
    async def clear_cache_async(self) -> Any:
        """Clear all caches asynchronously."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear Redis cache
            if self.redis_client:
                pattern = "async_nlp:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info("Async NLP cache cleared")
            
        except Exception as e:
            logger.error(f"Async cache clear error: {e}")


# Global async NLP processor instance
async_nlp_processor = AsyncNLPProcessor()


def get_async_nlp_processor() -> AsyncNLPProcessor:
    """Get global async NLP processor instance."""
    return async_nlp_processor


def async_nlp_decorator(func: Callable) -> Callable:
    """Decorator for async NLP enhancement."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Extract text from function arguments
        text = None
        for arg in args:
            if isinstance(arg, str):
                text = arg
                break
        
        if not text:
            for value in kwargs.values():
                if isinstance(value, str):
                    text = value
                    break
        
        if text:
            # Enhance text with async NLP
            enhanced = await async_nlp_processor.enhance_post_async(text)
            return enhanced
        
        return await func(*args, **kwargs)
    
    return wrapper 