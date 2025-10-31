from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import hashlib
import json
from typing import Dict, List, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
import spacy
from transformers import pipeline
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT
import orjson
import aioredis
from cachetools import TTLCache
import language_tool_python
from ...shared.logging import get_logger
from .seo_analyser import analyse_seo
from .executor_pool import get_pool, get_semaphore
from typing import Any, List, Dict, Optional
import logging
"""
Fast NLP Enhancer for LinkedIn Posts
===================================

Ultra-fast NLP-based quality enhancement with caching, parallel processing,
and optimizations for maximum speed.
"""


# Fast NLP libraries

# Performance libraries

# Grammar and style library


# Global executor and semaphore

logger = get_logger(__name__)


class FastNLPEnhancer:
    """
    Ultra-fast NLP enhancer with caching and parallel processing.
    
    Features:
    - Multi-layer caching (L1: Memory, L2: Redis)
    - Parallel processing of NLP tasks
    - Optimized model loading
    - Batch processing capabilities
    - Async operations
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize fast NLP enhancer."""
        self.redis_url = redis_url
        self.redis_client = None
        
        # L1: Memory cache
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)
        
        # Shared thread pool
        self.thread_pool = get_pool()
        
        # Shared semaphore for batch operations
        self.semaphore = get_semaphore()
        
        # Initialize models (lazy loading)
        self._models_loaded = False
        self.nlp = None
        self.sentiment = None
        self.keybert = None
        self.rewriter = None
        
        # Model load lock
        self._load_lock = asyncio.Lock()
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "average_processing_time": 0.0,
        }
        
        # Initialize Redis connection
        asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection."""
        try:
            self.redis_client = aioredis.from_url(
                self.redis_url,
                max_connections=20,
                decode_responses=False,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            logger.info("Fast NLP Redis cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis for NLP: {e}")
            self.redis_client = None
    
    def _load_models(self) -> Any:
        """Lazy load NLP models."""
        if self._models_loaded:
            return

        # Ensure only one coroutine loads models
        if hasattr(self, "_loading") and self._loading:
            return  # another coroutine is loading

        # acquire lock
        if hasattr(self, "_load_lock"):
            if self._load_lock.locked():
                return  # already loading elsewhere

        # synchronous context (called inside thread), still check flag
        if self._models_loaded:
            return

        # mark loading
        self._loading = True

        logger.info("Loading NLP models...")
        
        # Load models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Load spaCy
            self.nlp = executor.submit(spacy.load, "en_core_web_sm").result()
            
            # Load sentiment analyzer
            self.sentiment = SentimentIntensityAnalyzer()
            
            # Load KeyBERT
            self.keybert = executor.submit(KeyBERT, 'all-MiniLM-L6-v2').result()
            
            # Load text rewriter
            self.rewriter = executor.submit(
                pipeline, "text2text-generation", model="google/flan-t5-base"
            ).result()
            
            # Load grammar tool
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
        
        self._models_loaded = True
        self._loading = False
        logger.info("NLP models loaded successfully")
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _cached_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Cached sentiment analysis."""
        if not self.sentiment:
            self._load_models()
        return self.sentiment.polarity_scores(text)
    
    @lru_cache(maxsize=1000)
    def _cached_readability_analysis(self, text: str) -> float:
        """Cached readability analysis."""
        return textstat.flesch_reading_ease(text)
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache."""
        try:
            # Check L1 cache
            if cache_key in self.memory_cache:
                self.metrics["cache_hits"] += 1
                return self.memory_cache[cache_key]
            
            # Check L2 cache (Redis)
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
            logger.error(f"Cache get error: {e}")
            return None
    
    async def _set_cache(self, cache_key: str, result: Dict[str, Any], ttl: int = 3600):
        """Set result in cache."""
        try:
            # Set in L1 cache
            self.memory_cache[cache_key] = result
            
            # Set in L2 cache (Redis)
            if self.redis_client:
                serialized = orjson.dumps(result)
                await self.redis_client.setex(cache_key, ttl, serialized)
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def _process_nlp_tasks_parallel(self, text: str) -> Dict[str, Any]:
        """Process NLP tasks in parallel."""
        loop = asyncio.get_event_loop()
        
        # Create tasks for parallel processing
        tasks = [
            loop.run_in_executor(self.thread_pool, self._analyze_entities, text),
            loop.run_in_executor(self.thread_pool, self._extract_keywords, text),
            loop.run_in_executor(self.thread_pool, self._cached_sentiment_analysis, text),
            loop.run_in_executor(self.thread_pool, self._cached_readability_analysis, text),
            loop.run_in_executor(self.thread_pool, self._rewrite_text, text),
            loop.run_in_executor(self.thread_pool, self._grammar_check, text),
            loop.run_in_executor(self.thread_pool, analyse_seo, text),
        ]
        
        # Execute all tasks in parallel
        entities, keywords, sentiment, readability, rewritten, grammar, seo = await asyncio.gather(*tasks)
        
        seo_score = seo["seo_score"] if seo else 0.0
        
        grammar_issues, grammar_suggestions = grammar
        
        # Compute quality score
        quality_score = self._compute_quality_score(
            readability=readability,
            sentiment=sentiment.get("compound", 0),
            grammar_issues=len(grammar_issues)
        )
        
        # Compile result
        result = {
            "original": text,
            "rewritten": rewritten,
            "sentiment": sentiment,
            "entities": entities,
            "keywords": keywords,
            "readability": readability,
            "grammar_issues": grammar_issues,
            "grammar_suggestions": grammar_suggestions,
            "quality_score": quality_score,
            "seo_score": seo_score,
            "meta_description": seo.get("meta_description"),
        }
        
        # Cache result
        await self._set_cache(self._generate_cache_key(text), result)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(processing_time)
        
        return {
            "original": text,
            "enhanced": result,
            "processing_time": processing_time,
            "cached": False,
        }
    
    def _analyze_entities(self, text: str) -> List[tuple]:
        """Analyze entities in text."""
        if not self.nlp:
            self._load_models()
        
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        if not self.keybert:
            self._load_models()
        
        try:
            keywords = self.keybert.extract_keywords(text, top_n=5)
            return [kw[0] for kw in keywords]
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []
    
    def _rewrite_text(self, text: str) -> str:
        """Rewrite text for improvement."""
        if not self.rewriter:
            self._load_models()
        
        try:
            result = self.rewriter(f"Improve this LinkedIn post: {text}", max_length=256)
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Text rewriting error: {e}")
            return text
    
    def _grammar_check(self, text: str):
        """Perform grammar and style checks using language_tool_python."""
        if not hasattr(self, 'grammar_tool'):
            self._load_models()

        # Use LRU cache for repeated texts
        return self._cached_grammar_check(text)

    @lru_cache(maxsize=1000)
    def _cached_grammar_check(self, text: str):
        """Cached grammar check (thread-safe)."""
        try:
            # Quick heuristic: skip full analysis for very short texts
            if len(text) < 50:
                matches = self.grammar_tool.check(text)
            else:
                # Chunk the text to speed up processing
                chunks = text.split(". ")[:10]  # analyze first 10 sentences
                matches = []
                for chunk in chunks:
                    matches.extend(self.grammar_tool.check(chunk))

            issues = len(matches)
            suggestions = [f"{m.ruleIssueType}: {m.message}" for m in matches[:10]]
            return issues, suggestions
        except Exception as e:
            logger.error(f"Grammar check error: {e}")
            return 0, []
    
    async def enhance_post_fast(self, text: str) -> Dict[str, Any]:
        """
        Ultra-fast post enhancement with caching and parallel processing.
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
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
                }
            
            # Process NLP tasks in parallel
            nlp_results = await self._process_nlp_tasks_parallel(text)
            
            # Compute quality score
            quality_score = self._compute_quality_score(
                readability=nlp_results["readability"],
                sentiment=nlp_results["sentiment"].get("compound", 0),
                grammar_issues=len(nlp_results["grammar_issues"])
            )
            
            # Compile result
            result = {
                "original": text,
                "rewritten": nlp_results["rewritten"],
                "sentiment": nlp_results["sentiment"],
                "entities": nlp_results["entities"],
                "keywords": nlp_results["keywords"],
                "readability": nlp_results["readability"],
                "grammar_issues": nlp_results["grammar_issues"],
                "grammar_suggestions": nlp_results["grammar_suggestions"],
                "quality_score": quality_score,
                "seo_score": nlp_results["seo_score"],
                "meta_description": nlp_results.get("meta_description"),
            }
            
            # Cache result
            await self._set_cache(cache_key, result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time)
            
            return {
                "original": text,
                "enhanced": result,
                "processing_time": processing_time,
                "cached": False,
            }
            
        except Exception as e:
            logger.error(f"NLP enhancement error: {e}")
            processing_time = time.time() - start_time
            self._update_metrics(processing_time)
            
            return {
                "original": text,
                "enhanced": {"error": str(e)},
                "processing_time": processing_time,
                "cached": False,
            }
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics."""
        current_avg = self.metrics["average_processing_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    async def enhance_multiple_posts_fast(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Enhance multiple posts with batch processing."""
        chunk_size = 8
        results: List[Dict[str, Any]] = []

        async def enhance_single_post(text: str):
            
    """enhance_single_post function."""
async with self.semaphore:
                return await self.enhance_post_fast(text)

        for i in range(0, len(texts), chunk_size):
            batch = texts[i : i + chunk_size]
            tasks = [asyncio.create_task(enhance_single_post(t)) for t in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "original": texts[i],
                    "enhanced": {"error": str(result)},
                    "processing_time": 0,
                    "cached": False,
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
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
            "models_loaded": self._models_loaded,
        }
    
    async def clear_cache(self) -> Any:
        """Clear all caches."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear Redis cache
            if self.redis_client:
                pattern = "nlp_enhancement:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info("NLP cache cleared")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def _compute_quality_score(self, readability: float, sentiment: float, grammar_issues: int) -> float:
        """Compute overall quality score (0-100)."""
        # Base score from readability (scaled to 0-40)
        read_score = max(min((readability / 100) * 40, 40), 0)

        # Sentiment score contribution (scaled to 0-30)
        sent_score = max(min(((sentiment + 1) / 2) * 30, 30), 0)  # sentiment -1..1 to 0..30

        # Grammar penalty (up to -30)
        grammar_penalty = min(grammar_issues * 2, 30)  # 2 points penalty per issue

        quality = read_score + sent_score - grammar_penalty
        return max(min(quality, 100), 0)


# Global fast NLP enhancer instance
fast_nlp_enhancer = FastNLPEnhancer()


def get_fast_nlp_enhancer() -> FastNLPEnhancer:
    """Get global fast NLP enhancer instance."""
    return fast_nlp_enhancer


def fast_nlp_decorator(func: Callable) -> Callable:
    """Decorator for fast NLP enhancement."""
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
            # Enhance text with fast NLP
            enhanced = await fast_nlp_enhancer.enhance_post_fast(text)
            return enhanced
        
        return await func(*args, **kwargs)
    
    return wrapper 