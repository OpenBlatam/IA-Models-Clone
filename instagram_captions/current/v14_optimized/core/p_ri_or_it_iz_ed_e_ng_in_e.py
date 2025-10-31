from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple
from cachetools import TTLCache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from numba import jit
import numpy as np
import logging
import gc
import psutil
from contextlib import asynccontextmanager
import re
from ..types import OptimizedRequest, OptimizedResponse
from ..utils.prioritized_error_handling import (
from ..config import config
from typing import Any, List, Dict, Optional
"""
Instagram Captions API v14.0 - Prioritized Engine with Advanced Error Handling
Ultra-fast caption generation with prioritized error handling and comprehensive edge case coverage
"""


    prioritized_error_handler, prioritized_error_context, async_prioritized_error_context,
    ErrorCategory, ErrorPriority, PrioritizedError
)

logger = logging.getLogger(__name__)

class PrioritizedAIEngine:
    """Ultra-fast AI engine with prioritized error handling and edge case management"""
    
    def __init__(self) -> Any:
        """Initialize prioritized engine with comprehensive error handling"""
        self.device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        self.tokenizer = None
        self.model = None
        self.cache = TTLCache(maxsize=config.CACHE_SIZE, ttl=config.CACHE_TTL)
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        self.stats = {"requests": 0, "cache_hits": 0, "avg_time": 0.0, "errors": 0}
        
        # Performance optimizations with error handling
        if config.MIXED_PRECISION: 
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception as e:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.SYSTEM,
                    priority=ErrorPriority.MEDIUM,
                    message=f"Failed to initialize mixed precision scaler: {e}",
                    exception=e
                )
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Start model initialization
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self) -> Any:
        """Initialize models with comprehensive error handling"""
        async with async_prioritized_error_context(
            "model_initialization", 
            ErrorCategory.AI_MODEL, 
            ErrorPriority.CRITICAL
        ):
            try:
                # Check system resources before loading
                if not self.resource_monitor.check_resources():
                    raise ResourceError("Insufficient system resources for model loading")
                
                self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_NAME,
                    torch_dtype=torch.float16 if config.MIXED_PRECISION else torch.float32
                ).to(self.device)
                
                # JIT optimization with error handling
                if config.ENABLE_JIT:
                    try:
                        self.model = torch.jit.optimize_for_inference(self.model)
                    except Exception as e:
                        prioritized_error_handler.create_error(
                            category=ErrorCategory.AI_MODEL,
                            priority=ErrorPriority.MEDIUM,
                            message=f"JIT optimization failed: {e}",
                            exception=e
                        )
                
                logger.info(f"Models loaded successfully on {self.device}")
                
            except Exception as e:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.AI_MODEL,
                    priority=ErrorPriority.CRITICAL,
                    message=f"Model initialization failed: {e}",
                    exception=e
                )
                raise
    
    @jit(nopython=True)
    def _calculate_quality_score(self, caption: str, content: str) -> float:
        """JIT-optimized quality calculation with error handling"""
        try:
            caption_len = len(caption)
            content_len = len(content)
            word_count = len(caption.split())
            
            # Optimized scoring algorithm
            length_score = min(caption_len / 200.0, 1.0) * 30
            word_score = min(word_count / 20.0, 1.0) * 40
            relevance_score = 30.0
            
            return min(length_score + word_score + relevance_score, 100.0)
        except Exception:
            # Fallback score if calculation fails
            return 50.0
    
    def _generate_cache_key(self, request: OptimizedRequest) -> str:
        """Generate secure cache key with validation"""
        try:
            key_data = f"{request.content_description}:{request.style}:{request.hashtag_count}"
            return hashlib.sha256(key_data.encode()).hexdigest()
        except Exception as e:
            prioritized_error_handler.create_error(
                category=ErrorCategory.CACHE,
                priority=ErrorPriority.LOW,
                message=f"Cache key generation failed: {e}",
                exception=e
            )
            # Fallback cache key
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    async def generate_caption(self, request: OptimizedRequest, request_id: str) -> OptimizedResponse:
        """Ultra-fast caption generation with prioritized error handling"""
        start_time = time.time()
        if not request or not request_id:
            msg = "Missing or invalid request. Please provide all required fields."
            logger.error(f"[generate_caption] {msg} | request_id={request_id}")
            err = error_factory(
                "validation",
                "Missing request or request_id in generate_caption",
                msg,
                request_id=request_id
            )
            prioritized_error_handler.create_error(
                category=ErrorCategory.VALIDATION,
                priority=ErrorPriority.HIGH,
                message=err.args[0],
                request_id=request_id,
                context={"user_message": err.user_message, "code": err.code}
            )
            return OptimizedResponse(
                request_id=request_id or "unknown",
                caption=err.user_message,
                hashtags=[],
                quality_score=0.0,
                processing_time=time.time() - start_time,
                cache_hit=False,
                optimization_level=err.code
            )
        if not self.resource_monitor.check_resources():
            msg = "System resources are temporarily unavailable. Please try again later."
            logger.error(f"[generate_caption] {msg} | request_id={request_id}")
            err = error_factory(
                "resource",
                "Insufficient resources for caption generation",
                msg,
                request_id=request_id
            )
            prioritized_error_handler.create_error(
                category=ErrorCategory.RESOURCE,
                priority=ErrorPriority.HIGH,
                message=err.args[0],
                request_id=request_id,
                context={"user_message": err.user_message, "code": err.code}
            )
            return OptimizedResponse(
                request_id=request_id,
                caption=err.user_message,
                hashtags=[],
                quality_score=0.0,
                processing_time=time.time() - start_time,
                cache_hit=False,
                optimization_level=err.code
            )
        async with async_prioritized_error_context(
            "caption_generation", 
            ErrorCategory.SYSTEM, 
            ErrorPriority.HIGH,
            request_id
        ):
            try:
                validation_result = await self._validate_request(request, request_id)
                if not validation_result[0]:
                    logger.warning(f"[generate_caption] Validation failed: {validation_result[1]} | request_id={request_id}")
                    err = error_factory(
                        "validation",
                        f"Validation failed: {validation_result[1]}",
                        "Invalid input. Please check your request and try again.",
                        request_id=request_id,
                        context={"validation_errors": validation_result[1]}
                    )
                    prioritized_error_handler.create_error(
                        category=ErrorCategory.VALIDATION,
                        priority=ErrorPriority.MEDIUM,
                        message=err.args[0],
                        request_id=request_id,
                        context={"user_message": err.user_message, "code": err.code}
                    )
                    return await self._handle_validation_error(request, validation_result[1], request_id)
                security_result = await self._security_scan(request, request_id)
                if not security_result[0]:
                    logger.warning(f"[generate_caption] Security threat: {security_result[1]} | request_id={request_id}")
                    err = error_factory(
                        "security",
                        f"Security threat detected: {security_result[1]}",
                        "Request blocked due to security concerns.",
                        request_id=request_id,
                        context={"threats": security_result[1]}
                    )
                    prioritized_error_handler.create_error(
                        category=ErrorCategory.SECURITY,
                        priority=ErrorPriority.CRITICAL,
                        message=err.args[0],
                        request_id=request_id,
                        context={"user_message": err.user_message, "code": err.code}
                    )
                    return await self._handle_security_error(request, security_result[1], request_id)
                cache_result = await self._check_cache(request, start_time, request_id)
                if cache_result:
                    return cache_result
            except Exception as e:
                self.stats["errors"] += 1
                logger.exception(f"[generate_caption] Unexpected error: {e} | request_id={request_id}")
                err = error_factory(
                    "system",
                    f"Caption generation failed: {e}",
                    "An unexpected error occurred. Please try again later.",
                    request_id=request_id
                )
                prioritized_error_handler.create_error(
                    category=ErrorCategory.SYSTEM,
                    priority=ErrorPriority.HIGH,
                    message=err.args[0],
                    exception=e,
                    request_id=request_id,
                    context={"user_message": err.user_message, "code": err.code}
                )
                return OptimizedResponse(
                    request_id=request_id,
                    caption=err.user_message,
                    hashtags=[],
                    quality_score=0.0,
                    processing_time=time.time() - start_time,
                    cache_hit=False,
                    optimization_level=err.code
                )
            # Happy path last
            caption = await self._generate_with_ai(request, request_id)
            hashtags = await self._generate_hashtags(request, caption, request_id)
            quality_score = self._calculate_quality_score(caption, request.content_description)
            response = OptimizedResponse(
                request_id=request_id,
                caption=caption,
                hashtags=hashtags,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                cache_hit=False,
                optimization_level=request.optimization_level
            )
            await self._cache_response(request, response, request_id)
            self._update_stats(response.processing_time, request_id)
            return response
    
    async async def _validate_request(self, request: OptimizedRequest, request_id: str) -> Tuple[bool, List[str]]:
        """Validate request with comprehensive checks"""
        async with async_prioritized_error_context(
            "request_validation", 
            ErrorCategory.VALIDATION, 
            ErrorPriority.MEDIUM,
            request_id
        ):
            errors = []
            if not request.content_description or len(request.content_description.strip()) < 3:
                errors.append("Content description must be at least 3 characters")
            if len(request.content_description) > 1000:
                errors.append("Content description must be less than 1000 characters")
            valid_styles = ["casual", "professional", "inspirational", "playful"]
            if request.style not in valid_styles:
                errors.append(f"Style must be one of: {', '.join(valid_styles)}")
            if not (5 <= request.hashtag_count <= 30):
                errors.append("Hashtag count must be between 5 and 30")
            valid_levels = ["ultra_fast", "balanced", "quality"]
            if request.optimization_level not in valid_levels:
                errors.append(f"Optimization level must be one of: {', '.join(valid_levels)}")
            if errors:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.VALIDATION,
                    priority=ErrorPriority.MEDIUM,
                    message=f"Validation failed: {', '.join(errors)}",
                    context={"validation_errors": errors},
                    request_id=request_id
                )
                return False, errors
            return True, []
    
    async def _security_scan(self, request: OptimizedRequest, request_id: str) -> Tuple[bool, List[str]]:
        """Security scan for malicious content"""
        async with async_prioritized_error_context(
            "security_scan", 
            ErrorCategory.SECURITY, 
            ErrorPriority.HIGH,
            request_id
        ):
            threats = []
            sql_patterns = [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(\b(OR|AND)\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"])"
            ]
            xss_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ]
            cmd_patterns = [
                r"(\b(cmd|command|exec|system|eval|subprocess)\b)",
                r"[;&|`$()]",
                r"(\b(rm|del|format|shutdown|reboot)\b)"
            ]
            content = request.content_description.lower()
            for pattern in sql_patterns + xss_patterns + cmd_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append(f"Malicious pattern detected: {pattern}")
            if threats:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.SECURITY,
                    priority=ErrorPriority.CRITICAL,
                    message=f"Security threat detected: {', '.join(threats)}",
                    context={"threats": threats},
                    request_id=request_id
                )
                return False, threats
            return True, []
    
    async def _check_cache(self, request: OptimizedRequest, start_time: float, request_id: str) -> Optional[OptimizedResponse]:
        """Check cache with error handling"""
        async with async_prioritized_error_context(
            "cache_check", 
            ErrorCategory.CACHE, 
            ErrorPriority.LOW,
            request_id
        ):
            try:
                if not config.ENABLE_CACHE:
                    return None
                
                cache_key = self._generate_cache_key(request)
                if cache_key in self.cache:
                    cached_response = self.cache[cache_key]
                    self.stats["cache_hits"] += 1
                    return OptimizedResponse(
                        **cached_response,
                        cache_hit=True,
                        processing_time=time.time() - start_time
                    )
                
                return None
                
            except Exception as e:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.CACHE,
                    priority=ErrorPriority.LOW,
                    message=f"Cache check failed: {e}",
                    exception=e,
                    request_id=request_id
                )
                return None
    
    async def _generate_with_ai(self, request: OptimizedRequest, request_id: str) -> str:
        """Generate caption using optimized AI with comprehensive error handling"""
        if not self.model or not self.tokenizer:
            msg = "AI model is not available. Using fallback caption."
            logger.error(f"[_generate_with_ai] {msg} | request_id={request_id}")
            err = error_factory(
                "model",
                "AI models not initialized, using fallback",
                msg,
                request_id=request_id
            )
            prioritized_error_handler.create_error(
                category=ErrorCategory.AI_MODEL,
                priority=ErrorPriority.CRITICAL,
                message=err.args[0],
                request_id=request_id,
                context={"user_message": err.user_message, "code": err.code}
            )
            return self._fallback_generation(request)
        async with async_prioritized_error_context(
            "ai_generation", 
            ErrorCategory.AI_MODEL, 
            ErrorPriority.HIGH,
            request_id
        ):
            try:
                if self.device == "cuda" and not self._check_gpu_memory():
                    msg = "GPU memory exhausted. Switching to CPU fallback."
                    logger.error(f"[_generate_with_ai] {msg} | request_id={request_id}")
                    err = error_factory(
                        "resource",
                        "GPU memory exhausted, switching to CPU",
                        msg,
                        request_id=request_id
                    )
                    prioritized_error_handler.create_error(
                        category=ErrorCategory.MEMORY,
                        priority=ErrorPriority.HIGH,
                        message=err.args[0],
                        request_id=request_id,
                        context={"user_message": err.user_message, "code": err.code}
                    )
                    return self._fallback_generation(request)
                prompt = f"Write a {request.style} Instagram caption about {request.content_description}:"
                try:
                    inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=100, truncation=True)
                    inputs = inputs.to(self.device)
                except Exception as e:
                    logger.error(f"[_generate_with_ai] Tokenization failed: {e} | request_id={request_id}")
                    err = error_factory(
                        "model",
                        f"Tokenization failed: {e}",
                        "AI model could not process the request. Using fallback.",
                        request_id=request_id
                    )
                    prioritized_error_handler.create_error(
                        category=ErrorCategory.AI_MODEL,
                        priority=ErrorPriority.MEDIUM,
                        message=err.args[0],
                        exception=e,
                        request_id=request_id,
                        context={"user_message": err.user_message, "code": err.code}
                    )
                    return self._fallback_generation(request)
                with torch.no_grad():
                    try:
                        if config.MIXED_PRECISION:
                            with torch.cuda.amp.autocast():
                                outputs = self.model.generate(
                                    inputs,
                                    max_length=150,
                                    temperature=0.8,
                                    top_p=0.9,
                                    do_sample=True,
                                    pad_token_id=self.tokenizer.eos_token_id
                                )
                        else:
                            outputs = self.model.generate(
                                inputs,
                                max_length=150,
                                temperature=0.8,
                                top_p=0.9,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                    except Exception as e:
                        logger.error(f"[_generate_with_ai] Model generation failed: {e} | request_id={request_id}")
                        err = error_factory(
                            "model",
                            f"Model generation failed: {e}",
                            "AI model could not generate a caption. Using fallback.",
                            request_id=request_id
                        )
                        prioritized_error_handler.create_error(
                            category=ErrorCategory.AI_MODEL,
                            priority=ErrorPriority.HIGH,
                            message=err.args[0],
                            exception=e,
                            request_id=request_id,
                            context={"user_message": err.user_message, "code": err.code}
                        )
                        return self._fallback_generation(request)
                try:
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    caption = generated_text.replace(prompt, "").strip()
                except Exception as e:
                    logger.error(f"[_generate_with_ai] Text decoding failed: {e} | request_id={request_id}")
                    err = error_factory(
                        "model",
                        f"Text decoding failed: {e}",
                        "AI model could not decode the result. Using fallback.",
                        request_id=request_id
                    )
                    prioritized_error_handler.create_error(
                        category=ErrorCategory.AI_MODEL,
                        priority=ErrorPriority.MEDIUM,
                        message=err.args[0],
                        exception=e,
                        request_id=request_id,
                        context={"user_message": err.user_message, "code": err.code}
                    )
                    return self._fallback_generation(request)
                return caption or self._fallback_generation(request)
            except Exception as e:
                logger.exception(f"[_generate_with_ai] Unexpected error: {e} | request_id={request_id}")
                err = error_factory(
                    "model",
                    f"AI generation failed: {e}",
                    "AI model failed unexpectedly. Using fallback.",
                    request_id=request_id
                )
                prioritized_error_handler.create_error(
                    category=ErrorCategory.AI_MODEL,
                    priority=ErrorPriority.HIGH,
                    message=err.args[0],
                    exception=e,
                    request_id=request_id,
                    context={"user_message": err.user_message, "code": err.code}
                )
                return self._fallback_generation(request)
    
    def _fallback_generation(self, request: OptimizedRequest) -> str:
        """Fallback caption generation with error handling"""
        try:
            templates = {
                "casual": f"Just captured this amazing moment! {request.content_description} ✨",
                "professional": f"Professional insight: {request.content_description} #expertise",
                "inspirational": f"Inspired by {request.content_description} 🌟",
                "playful": f"Having fun with {request.content_description} 😄"
            }
            return templates.get(request.style, templates["casual"])
        except Exception as e:
            prioritized_error_handler.create_error(
                category=ErrorCategory.SYSTEM,
                priority=ErrorPriority.MEDIUM,
                message=f"Fallback generation failed: {e}",
                exception=e
            )
            return f"Amazing content: {request.content_description} ✨"
    
    async def _generate_hashtags(self, request: OptimizedRequest, caption: str, request_id: str) -> List[str]:
        """Generate optimized hashtags with error handling"""
        async with async_prioritized_error_context(
            "hashtag_generation", 
            ErrorCategory.SYSTEM, 
            ErrorPriority.LOW,
            request_id
        ):
            try:
                words = (request.content_description + " " + caption).lower().split()
                hashtags = []
                
                # Popular base hashtags
                base_hashtags = ["#instagram", "#love", "#instagood", "#photooftheday", "#beautiful"]
                
                # Content-specific hashtags with validation
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        hashtags.append(f"#{word}")
                
                # Combine and deduplicate
                all_hashtags = base_hashtags + hashtags
                unique_hashtags = list(dict.fromkeys(all_hashtags))
                
                return unique_hashtags[:request.hashtag_count]
                
            except Exception as e:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.SYSTEM,
                    priority=ErrorPriority.LOW,
                    message=f"Hashtag generation failed: {e}",
                    exception=e,
                    request_id=request_id
                )
                # Return safe fallback hashtags
                return ["#instagram", "#love", "#instagood", "#photooftheday", "#beautiful"][:request.hashtag_count]
    
    async def _cache_response(self, request: OptimizedRequest, response: OptimizedResponse, request_id: str):
        """Cache response with error handling"""
        async with async_prioritized_error_context(
            "cache_response", 
            ErrorCategory.CACHE, 
            ErrorPriority.LOW,
            request_id
        ):
            try:
                if config.ENABLE_CACHE:
                    cache_key = self._generate_cache_key(request)
                    self.cache[cache_key] = response.dict()
            except Exception as e:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.CACHE,
                    priority=ErrorPriority.LOW,
                    message=f"Cache storage failed: {e}",
                    exception=e,
                    request_id=request_id
                )
    
    def _update_stats(self, processing_time: float, request_id: str):
        """Update statistics with error handling"""
        try:
            self.stats["requests"] += 1
            self.stats["avg_time"] = (self.stats["avg_time"] * (self.stats["requests"] - 1) + processing_time) / self.stats["requests"]
        except Exception as e:
            prioritized_error_handler.create_error(
                category=ErrorCategory.SYSTEM,
                priority=ErrorPriority.LOW,
                message=f"Stats update failed: {e}",
                exception=e,
                request_id=request_id
            )
    
    async def _handle_validation_error(self, request: OptimizedRequest, errors: List[str], request_id: str) -> OptimizedResponse:
        """Handle validation errors with fallback"""
        prioritized_error_handler.create_error(
            category=ErrorCategory.VALIDATION,
            priority=ErrorPriority.MEDIUM,
            message=f"Validation errors: {', '.join(errors)}",
            context={"validation_errors": errors},
            request_id=request_id
        )
        return await self._generate_fallback_caption(request, request_id)
    
    async def _handle_security_error(self, request: OptimizedRequest, threats: List[str], request_id: str) -> OptimizedResponse:
        """Handle security errors with safe fallback"""
        prioritized_error_handler.create_error(
            category=ErrorCategory.SECURITY,
            priority=ErrorPriority.CRITICAL,
            message=f"Security threats: {', '.join(threats)}",
            context={"threats": threats},
            request_id=request_id
        )
        return await self._generate_fallback_caption(request, request_id)
    
    async def _generate_fallback_caption(self, request: OptimizedRequest, request_id: str) -> OptimizedResponse:
        """Generate fallback caption when main generation fails"""
        start_time = time.time()
        
        try:
            caption = self._fallback_generation(request)
            hashtags = ["#fallback", "#instagram", "#content", "#post", "#social"]
            
            response = OptimizedResponse(
                request_id=request_id,
                caption=caption,
                hashtags=hashtags,
                quality_score=30.0,
                processing_time=time.time() - start_time,
                cache_hit=False,
                optimization_level="balanced"
            )
            
            self._update_stats(response.processing_time, request_id)
            return response
            
        except Exception as e:
            prioritized_error_handler.create_error(
                category=ErrorCategory.SYSTEM,
                priority=ErrorPriority.HIGH,
                message=f"Fallback caption generation failed: {e}",
                exception=e,
                request_id=request_id
            )
            # Ultimate fallback
            return OptimizedResponse(
                request_id=request_id,
                caption=f"Content about {request.content_description}",
                hashtags=["#content", "#post"],
                quality_score=10.0,
                processing_time=time.time() - start_time,
                cache_hit=False,
                optimization_level="balanced"
            )
    
    def _check_gpu_memory(self) -> bool:
        """Check GPU memory availability"""
        try:
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                memory_total = torch.cuda.get_device_properties(0).total_memory
                
                # If more than 90% of GPU memory is used, return False
                return (memory_allocated + memory_reserved) / memory_total < 0.9
            return True
        except Exception:
            return True
    
    async def batch_generate(self, requests: List[OptimizedRequest], batch_id: str) -> List[OptimizedResponse]:
        """Batch processing with comprehensive error handling"""
        async with async_prioritized_error_context(
            "batch_generation", 
            ErrorCategory.BATCH_PROCESSING, 
            ErrorPriority.HIGH,
            batch_id
        ):
            try:
                # Validate batch size
                if len(requests) > 100:
                    prioritized_error_handler.create_error(
                        category=ErrorCategory.BATCH_PROCESSING,
                        priority=ErrorPriority.HIGH,
                        message=f"Batch size too large: {len(requests)}",
                        request_id=batch_id
                    )
                    raise ValueError(f"Batch size cannot exceed 100, got {len(requests)}")
                
                # Early return for non-batching mode
                if not config.ENABLE_BATCHING:
                    return [await self.generate_caption(req, f"{batch_id}-{i}") for i, req in enumerate(requests)]
                
                # Process in batches with error handling
                batch_size = config.BATCH_SIZE
                results = []
                
                for i in range(0, len(requests), batch_size):
                    batch = requests[i:i + batch_size]
                    try:
                        batch_results = await asyncio.gather(
                            *[self.generate_caption(req, f"{batch_id}-{i+j}") for j, req in enumerate(batch)],
                            return_exceptions=True
                        )
                        
                        # Handle individual batch failures
                        for j, result in enumerate(batch_results):
                            if isinstance(result, Exception):
                                prioritized_error_handler.create_error(
                                    category=ErrorCategory.BATCH_PROCESSING,
                                    priority=ErrorPriority.MEDIUM,
                                    message=f"Batch item {i+j} failed: {result}",
                                    exception=result,
                                    request_id=f"{batch_id}-{i+j}"
                                )
                                # Create fallback response
                                fallback_response = await self._generate_fallback_caption(batch[j], f"{batch_id}-{i+j}")
                                results.append(fallback_response)
                            else:
                                results.append(result)
                                
                    except Exception as e:
                        prioritized_error_handler.create_error(
                            category=ErrorCategory.BATCH_PROCESSING,
                            priority=ErrorPriority.HIGH,
                            message=f"Batch processing failed: {e}",
                            exception=e,
                            request_id=batch_id
                        )
                        # Create fallback responses for entire batch
                        for j, req in enumerate(batch):
                            fallback_response = await self._generate_fallback_caption(req, f"{batch_id}-{i+j}")
                            results.append(fallback_response)
                
                return results
                
            except Exception as e:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.BATCH_PROCESSING,
                    priority=ErrorPriority.HIGH,
                    message=f"Batch generation failed: {e}",
                    exception=e,
                    request_id=batch_id
                )
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics with error metrics"""
        try:
            error_stats = prioritized_error_handler.get_error_statistics()
            return {
                "total_requests": self.stats["requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["requests"], 1) * 100,
                "average_processing_time": self.stats["avg_time"],
                "cache_size": len(self.cache),
                "device": str(self.device),
                "error_rate": self.stats["errors"] / max(self.stats["requests"], 1) * 100,
                "optimizations_enabled": {
                    "jit": config.ENABLE_JIT,
                    "cache": config.ENABLE_CACHE,
                    "batching": config.ENABLE_BATCHING,
                    "mixed_precision": config.MIXED_PRECISION
                },
                "error_metrics": error_stats,
                "resource_metrics": self.resource_monitor.get_metrics()
            }
        except Exception as e:
            prioritized_error_handler.create_error(
                category=ErrorCategory.SYSTEM,
                priority=ErrorPriority.LOW,
                message=f"Stats generation failed: {e}",
                exception=e
            )
            return {"error": "Stats generation failed"}

class ResourceMonitor:
    """Resource monitoring and management"""
    
    def __init__(self) -> Any:
        self.memory_threshold = 0.9  # 90%
        self.cpu_threshold = 0.8     # 80%
        self.disk_threshold = 0.95   # 95%
    
    def check_resources(self) -> bool:
        """Check if system resources are sufficient"""
        try:
            # Check memory
            memory_usage = psutil.virtual_memory().percent / 100
            if memory_usage > self.memory_threshold:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.MEMORY,
                    priority=ErrorPriority.HIGH,
                    message=f"Memory usage too high: {memory_usage:.2%}"
                )
                return False
            
            # Check CPU
            cpu_usage = psutil.cpu_percent() / 100
            if cpu_usage > self.cpu_threshold:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.RESOURCE,
                    priority=ErrorPriority.MEDIUM,
                    message=f"CPU usage too high: {cpu_usage:.2%}"
                )
                return False
            
            # Check disk space
            disk_usage = psutil.disk_usage('/').percent / 100
            if disk_usage > self.disk_threshold:
                prioritized_error_handler.create_error(
                    category=ErrorCategory.RESOURCE,
                    priority=ErrorPriority.CRITICAL,
                    message=f"Disk space too low: {disk_usage:.2%}"
                )
                return False
            
            return True
            
        except Exception as e:
            prioritized_error_handler.create_error(
                category=ErrorCategory.SYSTEM,
                priority=ErrorPriority.MEDIUM,
                message=f"Resource check failed: {e}",
                exception=e
            )
            return True  # Allow operation if monitoring fails
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        try:
            return {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage('/').percent,
                "memory_available": psutil.virtual_memory().available,
                "cpu_count": psutil.cpu_count()
            }
        except Exception as e:
            prioritized_error_handler.create_error(
                category=ErrorCategory.SYSTEM,
                priority=ErrorPriority.LOW,
                message=f"Metrics collection failed: {e}",
                exception=e
            )
            return {"error": "Metrics collection failed"}

# Global prioritized engine instance
prioritized_engine = PrioritizedAIEngine()

# Custom exception classes
class ResourceError(Exception):
    """Resource-related exception"""
    pass

class SecurityError(Exception):
    """Security-related exception"""
    pass

class ValidationError(Exception):
    """Validation-related exception"""
    pass

class AIModelError(Exception):
    """AI model-related exception"""
    pass 

class CaptionAPIError(Exception):
    """Base exception for all caption API errors."""
    def __init__(self, message: str, user_message: str = None, code: str = None, request_id: str = None, context: dict = None):
        
    """__init__ function."""
super().__init__(message)
        self.user_message = user_message or "An error occurred."
        self.code = code or "error"
        self.request_id = request_id
        self.context = context or {}

class ValidationAPIError(CaptionAPIError):
    pass
class ResourceAPIError(CaptionAPIError):
    pass
class SecurityAPIError(CaptionAPIError):
    pass
class ModelAPIError(CaptionAPIError):
    pass
class SystemAPIError(CaptionAPIError):
    pass

def error_factory(error_type: str, message: str, user_message: str, request_id: str = None, context: dict = None) -> CaptionAPIError:
    if error_type == "validation":
        return ValidationAPIError(message, user_message, code="validation_error", request_id=request_id, context=context)
    if error_type == "resource":
        return ResourceAPIError(message, user_message, code="resource_error", request_id=request_id, context=context)
    if error_type == "security":
        return SecurityAPIError(message, user_message, code="security_error", request_id=request_id, context=context)
    if error_type == "model":
        return ModelAPIError(message, user_message, code="model_error", request_id=request_id, context=context)
    return SystemAPIError(message, user_message, code="system_error", request_id=request_id, context=context) 