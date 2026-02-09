from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import hashlib
import logging
import gc
import weakref
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict, deque
import psutil
import os
    import redis.asyncio as redis
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import numpy as np
    from numba import jit, cuda
    from prometheus_client import Counter, Gauge, Histogram, Summary
    import ray
from .models import (
from typing import Any, List, Dict, Optional
"""
Ultra-Optimized Copywriting Engine v2.0
=======================================

Advanced copywriting engine with:
- GPU acceleration and model quantization
- Intelligent caching with Redis
- Async batch processing
- Real-time optimization
- Advanced monitoring and metrics
- Memory optimization
- Security features
"""


# Advanced libraries
try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    NUMPY_AVAILABLE = True
    NUMBA_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    NUMBA_AVAILABLE = False

try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Local imports
    CopywritingRequest, CopywritingResponse, CopywritingVariant,
    PerformanceMetrics, RequestDict, ResponseDict
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UltraEngineConfig:
    """Ultra-optimized engine configuration"""
    # Performance settings
    max_workers: int = 8
    max_batch_size: int = 64
    cache_ttl: int = 7200  # 2 hours
    max_cache_size: int = 50000
    
    # GPU and optimization settings
    enable_gpu: bool = True
    enable_quantization: bool = True
    enable_mixed_precision: bool = True
    enable_batching: bool = True
    enable_caching: bool = True
    enable_ray_distributed: bool = False
    
    # Memory optimization
    enable_memory_optimization: bool = True
    max_memory_usage: float = 0.8  # 80% of available RAM
    gc_threshold: int = 1000
    
    # Monitoring settings
    enable_metrics: bool = True
    enable_profiling: bool = True
    enable_memory_monitoring: bool = True
    
    # Model settings
    default_model: str = "gpt2-medium"
    fallback_model: str = "distilgpt2"
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Cache settings
    redis_url: str = "redis://localhost:6379"
    cache_prefix: str = "copywriting:"
    enable_compression: bool = True
    
    # Timeout settings
    request_timeout: float = 45.0
    batch_timeout: float = 120.0
    
    # Security settings
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    enable_input_validation: bool = True


class MemoryManager:
    """Advanced memory management"""
    
    def __init__(self, config: UltraEngineConfig):
        
    """__init__ function."""
self.config = config
        self.memory_threshold = psutil.virtual_memory().total * config.max_memory_usage
        self.gc_counter = 0
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        current_memory = psutil.virtual_memory().used
        return current_memory < self.memory_threshold
    
    def optimize_memory(self) -> Any:
        """Perform memory optimization"""
        self.gc_counter += 1
        
        if self.gc_counter >= self.config.gc_threshold:
            gc.collect()
            self.gc_counter = 0
            
        if not self.check_memory_usage():
            # Force garbage collection
            gc.collect()
            
            # Clear caches if available
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()


class GPUMemoryManager:
    """GPU memory management"""
    
    def __init__(self) -> Any:
        self.gpu_available = torch.cuda.is_available() if TORCH_AVAILABLE else False
        
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if not self.gpu_available:
            return {"available": False}
            
        return {
            "available": True,
            "total": torch.cuda.get_device_properties(0).total_memory,
            "allocated": torch.cuda.memory_allocated(0),
            "cached": torch.cuda.memory_reserved(0)
        }
    
    def optimize_gpu_memory(self) -> Any:
        """Optimize GPU memory usage"""
        if self.gpu_available:
            torch.cuda.empty_cache()


class AdvancedCache:
    """Advanced caching with compression and TTL"""
    
    def __init__(self, redis_url: str, prefix: str = "copywriting:", enable_compression: bool = True):
        
    """__init__ function."""
self.redis_url = redis_url
        self.prefix = prefix
        self.enable_compression = enable_compression
        self.redis_client = None
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0}
        
    async def initialize(self) -> Any:
        """Initialize Redis connection"""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
            
        try:
            full_key = f"{self.prefix}{key}"
            value = await self.redis_client.get(full_key)
            
            if value:
                self.cache_stats["hits"] += 1
                return json.loads(value.decode())
            else:
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        if not self.redis_client:
            return
            
        try:
            full_key = f"{self.prefix}{key}"
            serialized_value = json.dumps(value)
            await self.redis_client.setex(full_key, ttl, serialized_value)
            self.cache_stats["sets"] += 1
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return self.cache_stats.copy()


class BatchProcessor:
    """Advanced batch processing with optimization"""
    
    def __init__(self, config: UltraEngineConfig):
        
    """__init__ function."""
self.config = config
        self.batch_queue = asyncio.Queue()
        self.processing_batches = False
        self.batch_stats = {"processed": 0, "total_items": 0, "avg_time": 0.0}
        
    async def add_to_batch(self, request, future) -> Any:
        """Add request to batch queue"""
        await self.batch_queue.put((request, future))
        return future
    
    async def start_batch_processing(self, processor_func) -> Any:
        """Start batch processing loop"""
        self.processing_batches = True
        
        while self.processing_batches:
            try:
                batch = []
                futures = []
                
                # Collect batch
                while len(batch) < self.config.max_batch_size:
                    try:
                        request, future = await asyncio.wait_for(
                            self.batch_queue.get(), timeout=1.0
                        )
                        batch.append(request)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Process batch
                    start_time = time.time()
                    results = await processor_func(batch)
                    processing_time = time.time() - start_time
                    
                    # Update futures
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
                    
                    # Update stats
                    self.batch_stats["processed"] += 1
                    self.batch_stats["total_items"] += len(batch)
                    self.batch_stats["avg_time"] = (
                        (self.batch_stats["avg_time"] * (self.batch_stats["processed"] - 1) + processing_time) /
                        self.batch_stats["processed"]
                    )
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Mark futures as failed
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
    
    def stop_batch_processing(self) -> Any:
        """Stop batch processing"""
        self.processing_batches = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return self.batch_stats.copy()


class UltraCopywritingEngine:
    """Ultra-optimized copywriting engine"""
    
    def __init__(self, config: Optional[UltraEngineConfig] = None):
        
    """__init__ function."""
self.config = config or UltraEngineConfig()
        self.metrics = PerformanceMetrics()
        
        # Initialize managers
        self.memory_manager = MemoryManager(self.config)
        self.gpu_manager = GPUMemoryManager()
        self.cache = AdvancedCache(
            self.config.redis_url,
            self.config.cache_prefix,
            self.config.enable_compression
        )
        self.batch_processor = BatchProcessor(self.config)
        
        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        # Performance tracking
        self.request_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)
        self.active_requests = 0
        self.request_lock = threading.Lock()
        
        # Initialize Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.config.enable_metrics:
            self._init_prometheus_metrics()
        
        # Engine state
        self.is_initialized = False
        self.shutdown_event = threading.Event()
        
        # Initialize Ray if available
        if RAY_AVAILABLE and self.config.enable_ray_distributed:
            ray.init()
        
        logger.info("Ultra Copywriting Engine initialized")
    
    def _init_prometheus_metrics(self) -> Any:
        """Initialize Prometheus metrics"""
        self.prometheus_metrics = {
            'requests_total': Counter('ultra_copywriting_requests_total', 'Total requests'),
            'requests_duration': Histogram('ultra_copywriting_requests_duration_seconds', 'Request duration'),
            'cache_hits': Counter('ultra_copywriting_cache_hits_total', 'Cache hits'),
            'cache_misses': Counter('ultra_copywriting_cache_misses_total', 'Cache misses'),
            'errors_total': Counter('ultra_copywriting_errors_total', 'Total errors'),
            'active_requests': Gauge('ultra_copywriting_active_requests', 'Active requests'),
            'batch_size': Histogram('ultra_copywriting_batch_size', 'Batch size'),
            'optimization_applied': Counter('ultra_copywriting_optimization_applied_total', 'Optimizations applied'),
            'memory_usage': Gauge('ultra_copywriting_memory_usage_bytes', 'Memory usage'),
            'gpu_memory_usage': Gauge('ultra_copywriting_gpu_memory_usage_bytes', 'GPU memory usage'),
            'batch_processing_time': Histogram('ultra_copywriting_batch_processing_time_seconds', 'Batch processing time')
        }
    
    async def initialize(self) -> Any:
        """Initialize the engine"""
        try:
            logger.info("Initializing Ultra Copywriting Engine...")
            
            # Initialize cache
            await self.cache.initialize()
            
            # Load models
            await self._load_models()
            
            # Start background tasks
            asyncio.create_task(self._background_optimization_loop())
            asyncio.create_task(self._background_cleanup_loop())
            asyncio.create_task(self._background_monitoring_loop())
            
            # Start batch processing if enabled
            if self.config.enable_batching:
                asyncio.create_task(
                    self.batch_processor.start_batch_processing(self._process_batch)
                )
            
            self.is_initialized = True
            logger.info("Ultra Copywriting Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing engine: {e}")
            raise
    
    async def _load_models(self) -> Any:
        """Load ML models with optimization"""
        if TORCH_AVAILABLE:
            try:
                # Load models in background
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, self._load_torch_models
                )
                logger.info("ML models loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading ML models: {e}")
    
    def _load_torch_models(self) -> Any:
        """Load PyTorch models with optimization"""
        if TORCH_AVAILABLE:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.default_model)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.default_model)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to GPU if available
            if self.config.enable_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                
                # Enable mixed precision if requested
                if self.config.enable_mixed_precision:
                    self.model = self.model.half()
                
                logger.info("Model moved to GPU with optimizations")
            
            # Enable quantization if requested
            if self.config.enable_quantization:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Model quantization applied")
    
    async async def process_request(self, request) -> Dict[str, Any]:
        """Process copywriting request with ultra optimization"""
        start_time = time.time()
        request_id = self._generate_request_id(request)
        
        try:
            with self.request_lock:
                self.active_requests += 1
            
            # Memory optimization
            self.memory_manager.optimize_memory()
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self.cache.get(cache_key)
            
            if cached_response:
                if PROMETHEUS_AVAILABLE:
                    self.prometheus_metrics['cache_hits'].inc()
                return cached_response
            
            if PROMETHEUS_AVAILABLE:
                self.prometheus_metrics['cache_misses'].inc()
            
            # Process request
            if self.config.enable_batching:
                # Use batch processing
                future = asyncio.Future()
                await self.batch_processor.add_to_batch(request, future)
                response = await asyncio.wait_for(future, timeout=self.config.request_timeout)
            else:
                # Direct processing
                response = await self._process_request_internal(request, request_id)
            
            # Cache response
            await self.cache.set(cache_key, response, self.config.cache_ttl)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, cache_hit=False)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            if PROMETHEUS_AVAILABLE:
                self.prometheus_metrics['errors_total'].inc()
            
            # Return fallback response
            return self._create_fallback_response(request, str(e))
            
        finally:
            with self.request_lock:
                self.active_requests -= 1
    
    async async def _process_request_internal(self, request, request_id: str) -> Dict[str, Any]:
        """Internal request processing"""
        # Generate content
        content = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, self._generate_content, request
        )
        
        # Generate variants
        variants = await self._generate_variants(request)
        
        # Apply optimizations
        variants = await self._apply_optimizations(variants, request)
        
        return {
            "request_id": request_id,
            "content": content,
            "variants": variants,
            "processing_time": time.time(),
            "model_used": self.config.default_model
        }
    
    async def _process_batch(self, requests) -> List[Dict[str, Any]]:
        """Process batch of requests"""
        start_time = time.time()
        
        try:
            # Process batch in parallel
            tasks = [
                self._process_request_internal(request, self._generate_request_id(request))
                for request in requests
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Batch processing error for request {i}: {response}")
                    processed_responses.append(
                        self._create_fallback_response(requests[i], str(response))
                    )
                else:
                    processed_responses.append(response)
            
            # Update batch metrics
            processing_time = time.time() - start_time
            if PROMETHEUS_AVAILABLE:
                self.prometheus_metrics['batch_processing_time'].observe(processing_time)
                self.prometheus_metrics['batch_size'].observe(len(requests))
            
            return processed_responses
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Return fallback responses for all requests
            return [
                self._create_fallback_response(request, str(e))
                for request in requests
            ]
    
    def _generate_content(self, request) -> str:
        """Generate content using optimized model"""
        try:
            if TORCH_AVAILABLE and hasattr(self, 'model'):
                return self._generate_with_torch(request)
            else:
                return self._generate_fallback(request)
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            return self._generate_fallback(request)
    
    def _generate_with_torch(self, request) -> str:
        """Generate content using PyTorch model"""
        prompt = self._build_prompt(request)
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        if self.config.enable_gpu and torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new content
        return generated_text[len(prompt):].strip()
    
    def _generate_fallback(self, request) -> str:
        """Fallback content generation"""
        prompt = self._build_prompt(request)
        
        # Simple template-based generation
        templates = {
            "instagram": f"✨ {prompt}\n\n#marketing #copywriting #content",
            "facebook": f"🚀 {prompt}\n\nWhat do you think? Share your thoughts below! 👇",
            "twitter": f"💡 {prompt}\n\n#marketing #tips",
            "linkedin": f"📈 {prompt}\n\nThis approach has helped many businesses achieve their goals.",
            "email": f"Subject: {prompt}\n\nDear [Name],\n\n{prompt}\n\nBest regards,\n[Your Name]"
        }
        
        return templates.get(request.get('platform', '').lower(), f"📝 {prompt}")
    
    def _build_prompt(self, request) -> str:
        """Build optimized prompt"""
        prompt = request.get('prompt', '')
        platform = request.get('platform', 'general')
        content_type = request.get('content_type', 'post')
        tone = request.get('tone', 'professional')
        target_audience = request.get('target_audience', 'general')
        keywords = request.get('keywords', [])
        brand_voice = request.get('brand_voice', 'professional')
        
        base_prompt = f"Generate {content_type} copy for {platform} platform"
        
        if tone:
            base_prompt += f" with {tone} tone"
        
        if target_audience:
            base_prompt += f" targeting {target_audience}"
        
        if keywords:
            base_prompt += f" including keywords: {', '.join(keywords)}"
        
        if brand_voice:
            base_prompt += f" in {brand_voice} brand voice"
        
        return f"{base_prompt}: {prompt}"
    
    async def _generate_variants(self, request) -> List[Dict[str, Any]]:
        """Generate content variants"""
        variants = []
        num_variants = request.get('num_variants', 3)
        
        # Generate multiple variants
        for i in range(min(3, num_variants)):
            variant_content = self._generate_content(request)
            
            variant = {
                "id": f"variant_{i+1}",
                "content": variant_content,
                "score": 0.8 - (i * 0.1),  # Decreasing score for variants
                "metadata": {
                    "generation_method": "torch" if TORCH_AVAILABLE else "fallback",
                    "variant_index": i
                }
            }
            variants.append(variant)
        
        return variants
    
    async def _apply_optimizations(self, variants, request) -> List[Dict[str, Any]]:
        """Apply content optimizations"""
        optimized_variants = []
        
        for variant in variants:
            # Calculate scores
            relevance_score = await self._calculate_relevance_score(variant, request)
            engagement_score = await self._calculate_engagement_score(variant, request)
            conversion_score = await self._calculate_conversion_score(variant, request)
            
            # Update variant with scores
            variant["score"] = (relevance_score + engagement_score + conversion_score) / 3
            variant["metadata"].update({
                "relevance_score": relevance_score,
                "engagement_score": engagement_score,
                "conversion_score": conversion_score
            })
            
            optimized_variants.append(variant)
        
        # Sort by score
        optimized_variants.sort(key=lambda x: x["score"], reverse=True)
        
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['optimization_applied'].inc()
        
        return optimized_variants
    
    async def _calculate_relevance_score(self, variant, request) -> float:
        """Calculate relevance score"""
        # Simple keyword matching
        keywords = request.get('keywords', [])
        if keywords:
            keyword_matches = sum(1 for keyword in keywords 
                                if keyword.lower() in variant["content"].lower())
            return min(1.0, keyword_matches / len(keywords))
        return 0.8
    
    async def _calculate_engagement_score(self, variant, request) -> float:
        """Calculate engagement score"""
        # Simple heuristics
        content_length = len(variant["content"])
        has_emoji = any(ord(char) > 127 for char in variant["content"])
        has_hashtag = '#' in variant["content"]
        has_question = '?' in variant["content"]
        
        score = 0.5  # Base score
        
        if 50 <= content_length <= 200:
            score += 0.2
        if has_emoji:
            score += 0.1
        if has_hashtag:
            score += 0.1
        if has_question:
            score += 0.1
        
        return min(1.0, score)
    
    async def _calculate_conversion_score(self, variant, request) -> float:
        """Calculate conversion score"""
        # Simple conversion indicators
        conversion_words = ['buy', 'get', 'start', 'try', 'learn', 'discover', 'save', 'offer']
        has_cta = any(word in variant["content"].lower() for word in conversion_words)
        
        return 0.9 if has_cta else 0.6
    
    def _generate_cache_key(self, request) -> str:
        """Generate cache key"""
        content = f"{request.get('prompt', '')}:{request.get('platform', '')}:{request.get('content_type', '')}:{request.get('tone', '')}:{request.get('target_audience', '')}:{','.join(request.get('keywords', []))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _generate_request_id(self, request) -> str:
        """Generate unique request ID"""
        timestamp = int(time.time() * 1000)
        return f"req_{timestamp}_{hash(request.get('prompt', '')) % 10000}"
    
    def _update_metrics(self, processing_time: float, cache_hit: bool = False, 
                       optimization_applied: bool = False, error: bool = False):
        """Update performance metrics"""
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['requests_total'].inc()
            self.prometheus_metrics['requests_duration'].observe(processing_time)
            self.prometheus_metrics['active_requests'].set(self.active_requests)
            
            # Update memory metrics
            memory_info = psutil.virtual_memory()
            self.prometheus_metrics['memory_usage'].set(memory_info.used)
            
            if self.gpu_manager.gpu_available:
                gpu_info = self.gpu_manager.get_gpu_memory_info()
                self.prometheus_metrics['gpu_memory_usage'].set(gpu_info.get('allocated', 0))
        
        # Update internal metrics
        self.request_history.append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'cache_hit': cache_hit,
            'error': error
        })
    
    def _create_fallback_response(self, request, error_message: str) -> Dict[str, Any]:
        """Create fallback response"""
        return {
            "request_id": self._generate_request_id(request),
            "content": f"Error generating content: {error_message}",
            "variants": [],
            "processing_time": time.time(),
            "model_used": "fallback",
            "error": error_message
        }
    
    async def _background_optimization_loop(self) -> Any:
        """Background optimization loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._perform_background_optimizations()
            except Exception as e:
                logger.error(f"Background optimization error: {e}")
    
    async def _background_cleanup_loop(self) -> Any:
        """Background cleanup loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                await self._perform_cleanup()
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def _background_monitoring_loop(self) -> Any:
        """Background monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute
                await self._perform_monitoring()
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
    
    async def _perform_background_optimizations(self) -> Any:
        """Perform background optimizations"""
        # Memory optimization
        self.memory_manager.optimize_memory()
        
        # GPU memory optimization
        self.gpu_manager.optimize_gpu_memory()
        
        # Model optimization (if needed)
        if TORCH_AVAILABLE and hasattr(self, 'model'):
            # Recompile model for better performance
            pass
    
    async def _perform_cleanup(self) -> Any:
        """Perform cleanup tasks"""
        # Clear old request history
        if len(self.request_history) > 5000:
            # Keep only recent 5000 requests
            self.request_history = deque(list(self.request_history)[-5000:], maxlen=5000)
        
        # Clear old performance history
        if len(self.performance_history) > 500:
            self.performance_history = deque(list(self.performance_history)[-500:], maxlen=500)
        
        # Force garbage collection
        gc.collect()
    
    async def _perform_monitoring(self) -> Any:
        """Perform monitoring tasks"""
        # Log system status
        memory_info = psutil.virtual_memory()
        gpu_info = self.gpu_manager.get_gpu_memory_info()
        
        logger.info(f"System Status - Memory: {memory_info.percent}%, "
                   f"Active Requests: {self.active_requests}, "
                   f"GPU Memory: {gpu_info.get('allocated', 0) / 1024**3:.2f}GB")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        memory_info = psutil.virtual_memory()
        gpu_info = self.gpu_manager.get_gpu_memory_info()
        cache_stats = self.cache.get_stats()
        batch_stats = self.batch_processor.get_stats()
        
        return {
            "is_initialized": self.is_initialized,
            "active_requests": self.active_requests,
            "total_requests": len(self.request_history),
            "memory_usage": {
                "total": memory_info.total,
                "used": memory_info.used,
                "percent": memory_info.percent
            },
            "gpu_memory": gpu_info,
            "cache_stats": cache_stats,
            "batch_stats": batch_stats,
            "performance": {
                "avg_processing_time": sum(r['processing_time'] for r in self.request_history) / len(self.request_history) if self.request_history else 0,
                "cache_hit_rate": cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0
            }
        }
    
    async def shutdown(self) -> Any:
        """Shutdown the engine"""
        logger.info("Shutting down Ultra Copywriting Engine...")
        
        self.shutdown_event.set()
        
        # Stop batch processing
        self.batch_processor.stop_batch_processing()
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Shutdown Ray if initialized
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
        
        logger.info("Ultra Copywriting Engine shutdown complete") 