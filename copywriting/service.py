from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging
import orjson
import aioredis
import httpx
import aiohttp
import structlog
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from models import (
    CopywritingRequest,
    CopywritingResponse,
    BatchCopywritingRequest,
    BatchCopywritingResponse,
    FeedbackRequest,
    TaskStatus,
    CopywritingInput,
    CopywritingOutput,
    Feedback,
    SectionFeedback,
    CopyVariantHistory,
    get_settings
)
import json
import hashlib
from typing import Any, List, Dict, Optional

# Import v11 optimized engine
from ultra_optimized_engine_v11 import UltraOptimizedEngineV11, get_engine, cleanup_engine

"""
Optimized Copywriting Service with High-Performance Libraries.

Enhanced service using orjson, asyncio, and caching for ultra-fast operations.
Now includes v11 integration for maximum performance.
"""


# High-performance imports
try:
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Fast HTTP client
try:
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Performance monitoring

# Import optimized models
from models import (
    CopywritingInput, CopywritingOutput, CopyVariant, Metric,
    CopyTone, ContentType, Platform, Language, BrandVoice,
    get_settings, validate_input_fast, calculate_metrics_fast
)

logger = structlog.get_logger(__name__)

# Performance decorator
def performance_monitor(func) -> Any:
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000
            logger.info(f"{func.__name__} completed", duration_ms=duration)
            return result
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            logger.error(f"{func.__name__} failed", duration_ms=duration, error=str(e))
            raise
    return wrapper

class OptimizedCopywritingService:
    """Ultra-optimized copywriting service with performance enhancements."""
    
    def __init__(self) -> Any:
        self.settings = get_settings()
        self.redis_client = None
        self.http_client = None
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, mp.cpu_count() * 4))
        
        # Performance templates cache
        self._template_cache = {}
        self._metrics_cache = {}
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._total_processing_time = 0.0
        
        # Initialize async components
        asyncio.create_task(self._initialize_async_components())
    
    async def _initialize_async_components(self) -> Any:
        """Initialize async components like Redis and HTTP client."""
        try:
            if REDIS_AVAILABLE:
                self.redis_client = await aioredis.from_url(
                    self.settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("Redis client initialized")
            
            if HTTPX_AVAILABLE:
                self.http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
                )
                logger.info("HTTP client initialized")
                
        except Exception as e:
            logger.warning(f"Failed to initialize async components: {e}")
    
    @performance_monitor
    async def generate_copy(self, input_data: CopywritingInput) -> CopywritingOutput:
        """Generate optimized copywriting content."""
        start_time = time.perf_counter()
        self._request_count += 1
        
        try:
            # Fast validation
            if not validate_input_fast(input_data.model_dump()):
                raise ValueError("Invalid input data")
            
            # Check cache first
            cache_key = self._generate_cache_key(input_data)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return CopywritingOutput.from_json(cached_result)
            
            # Generate variants in parallel
            variants = await self._generate_variants_parallel(input_data)
            
            # Calculate metrics for all variants
            await self._calculate_metrics_parallel(variants)
            
            # Select best variant
            best_variant_id = self._select_best_variant(variants)
            
            # Create output
            generation_time = time.perf_counter() - start_time
            self._total_processing_time += generation_time
            
            output = CopywritingOutput(
                variants=variants,
                model_used="optimized-service",
                generation_time=generation_time,
                best_variant_id=best_variant_id,
                confidence_score=0.85,
                tracking_id=input_data.tracking_id,
                created_at=datetime.now()
            )
            
            # Cache result
            await self._cache_result(cache_key, output.to_json())
            
            logger.info(f"Generated {len(variants)} variants", 
                       generation_time=generation_time,
                       best_variant=best_variant_id)
            
            return output
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Generation failed: {e}")
            raise
    
    async def _generate_variants_parallel(self, input_data: CopywritingInput) -> List[CopyVariant]:
        """Generate multiple variants in parallel."""
        tasks = []
        
        # Create generation tasks
        for i in range(input_data.max_variants):
            task = asyncio.create_task(
                self._generate_single_variant(input_data, variant_index=i)
            )
            tasks.append(task)
        
        # Execute in parallel
        variants = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_variants = [
            variant for variant in variants 
            if isinstance(variant, CopyVariant)
        ]
        
        return successful_variants
    
    async def _generate_single_variant(self, input_data: CopywritingInput, variant_index: int = 0) -> CopyVariant:
        """Generate a single copy variant with optimizations."""
        
        # Use template-based generation for speed
        template = self._get_optimized_template(input_data.content_type, input_data.tone)
        
        # Generate content based on template and input
        headline = await self._generate_headline(input_data, template, variant_index)
        primary_text = await self._generate_primary_text(input_data, template, variant_index)
        call_to_action = await self._generate_cta(input_data, variant_index)
        hashtags = await self._generate_hashtags(input_data)
        
        # Calculate basic metrics
        metrics = calculate_metrics_fast(f"{headline} {primary_text}")
        
        variant = CopyVariant(
            variant_id=f"{input_data.tracking_id or 'gen'}_{variant_index}_{int(time.time())}",
            headline=headline,
            primary_text=primary_text,
            call_to_action=call_to_action,
            hashtags=hashtags,
            character_count=metrics["character_count"],
            word_count=metrics["word_count"],
            created_at=datetime.now()
        )
        
        return variant
    
    def _get_optimized_template(self, content_type: ContentType, tone: CopyTone) -> Dict[str, str]:
        """Get optimized template from cache or generate."""
        cache_key = f"{content_type}_{tone}"
        
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # High-performance template generation
        templates = {
            ContentType.ad_copy: {
                CopyTone.urgent: {
                    "headline": "🚨 {product} - ¡Oferta Limitada!",
                    "text": "No pierdas esta oportunidad única. {product} te ayuda a {benefit}. ¡Solo por tiempo limitado!",
                    "cta": "¡Aprovecha Ahora!"
                },
                CopyTone.professional: {
                    "headline": "{product} - La Solución Profesional",
                    "text": "Descubre cómo {product} puede optimizar {benefit} para tu empresa.",
                    "cta": "Solicita Demo"
                },
                CopyTone.casual: {
                    "headline": "¿Conoces {product}?",
                    "text": "Te va a encantar {product}. Es perfecto para {benefit}.",
                    "cta": "Pruébalo Gratis"
                }
            },
            ContentType.social_post: {
                CopyTone.friendly: {
                    "headline": "¡Hola! 👋",
                    "text": "Quería compartir contigo {product}. Me ha ayudado mucho con {benefit}.",
                    "cta": "¿Qué opinas?"
                },
                CopyTone.inspirational: {
                    "headline": "✨ Transforma tu {target}",
                    "text": "Con {product}, puedes lograr {benefit} de manera increíble.",
                    "cta": "¡Empieza Hoy!"
                }
            },
            ContentType.email_subject: {
                CopyTone.urgent: {
                    "headline": "⏰ Solo quedan horas: {product}",
                    "text": "",
                    "cta": ""
                },
                CopyTone.professional: {
                    "headline": "{product} - Actualización Importante",
                    "text": "",
                    "cta": ""
                }
            }
        }
        
        template = templates.get(content_type, {}).get(tone, {
            "headline": "{product} para {target}",
            "text": "Descubre {product} y mejora {benefit}.",
            "cta": "Más Información"
        })
        
        self._template_cache[cache_key] = template
        return template
    
    async def _generate_headline(self, input_data: CopywritingInput, template: Dict[str, str], variant_index: int) -> str:
        """Generate optimized headline."""
        headline_template = template.get("headline", "{product}")
        
        # Extract product name from description
        product_name = input_data.product_description.split('.')[0][:50]
        
        # Add variation for different variants
        variations = ["", " 🔥", " ⭐", " 💎", " 🚀"]
        variation = variations[variant_index % len(variations)]
        
        headline = headline_template.format(product=product_name) + variation
        
        return headline[:200]  # Limit length
    
    async def _generate_primary_text(self, input_data: CopywritingInput, template: Dict[str, str], variant_index: int) -> str:
        """Generate optimized primary text."""
        text_template = template.get("text", "Descubre {product}.")
        
        # Enhanced text generation
        product_name = input_data.product_description.split('.')[0][:50]
        benefit = "optimizar tus procesos"
        
        if input_data.key_points:
            benefit = input_data.key_points[0] if input_data.key_points else benefit
        
        text = text_template.format(product=product_name, benefit=benefit)
        
        # Add key points if available
        if input_data.key_points and len(input_data.key_points) > 1:
            additional_points = " ".join([f"✓ {point}" for point in input_data.key_points[1:3]])
            text += f" {additional_points}"
        
        return text[:2000]  # Platform-specific limit
    
    async def _generate_cta(self, input_data: CopywritingInput, variant_index: int) -> str:
        """Generate optimized call-to-action."""
        cta_options = {
            CopyTone.urgent: ["¡Actúa Ahora!", "¡No Esperes Más!", "¡Solo Hoy!"],
            CopyTone.professional: ["Solicita Información", "Contáctanos", "Programa Demo"],
            CopyTone.casual: ["Pruébalo", "Échale un Vistazo", "Descúbrelo"],
            CopyTone.friendly: ["¡Te Encantará!", "¡Pruébalo!", "¡Compártelo!"]
        }
        
        options = cta_options.get(input_data.tone, ["Más Información"])
        return options[variant_index % len(options)]
    
    async def _generate_hashtags(self, input_data: CopywritingInput) -> List[str]:
        """Generate optimized hashtags."""
        hashtags = []
        
        # Platform-specific hashtag generation
        if input_data.target_platform in [Platform.instagram, Platform.twitter, Platform.tiktok]:
            # Extract keywords from product description
            words = input_data.product_description.lower().split()
            relevant_words = [word for word in words if len(word) > 3][:5]
            
            # Add platform-specific hashtags
            platform_hashtags = {
                Platform.instagram: ["#instagram", "#social", "#marketing"],
                Platform.twitter: ["#twitter", "#tech", "#innovation"],
                Platform.tiktok: ["#tiktok", "#viral", "#trending"]
            }
            
            hashtags.extend([f"#{word}" for word in relevant_words])
            hashtags.extend(platform_hashtags.get(input_data.target_platform, []))
            
            # Add key points as hashtags
            if input_data.key_points:
                for point in input_data.key_points[:3]:
                    clean_point = point.replace(" ", "").replace("-", "")[:20]
                    hashtags.append(f"#{clean_point}")
        
        return hashtags[:10]  # Limit number of hashtags
    
    async def _calculate_metrics_parallel(self, variants: List[CopyVariant]):
        """Calculate metrics for all variants in parallel."""
        tasks = []
        
        for variant in variants:
            task = asyncio.create_task(self._calculate_variant_metrics(variant))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _calculate_variant_metrics(self, variant: CopyVariant):
        """Calculate comprehensive metrics for a variant."""
        full_text = f"{variant.headline} {variant.primary_text}"
        
        # Basic metrics
        basic_metrics = calculate_metrics_fast(full_text)
        
        # Advanced metrics (simulated for performance)
        readability_score = min(100, max(0, 100 - len(full_text.split()) * 2))
        engagement_prediction = min(1.0, (readability_score / 100) * 0.8 + 0.2)
        
        # Create metrics objects
        metrics = [
            Metric(name="readability", value=readability_score),
            Metric(name="engagement_prediction", value=engagement_prediction),
            Metric(name="word_count", value=basic_metrics["word_count"]),
            Metric(name="character_count", value=basic_metrics["character_count"])
        ]
        
        # Update variant
        variant.evaluation_metrics = metrics
        variant.readability_score = readability_score
        variant.engagement_prediction = engagement_prediction
    
    def _select_best_variant(self, variants: List[CopyVariant]) -> str:
        """Select the best variant based on metrics."""
        if not variants:
            return ""
        
        # Simple scoring based on engagement prediction and readability
        best_variant = max(variants, key=lambda v: (
            (v.engagement_prediction or 0) * 0.6 +
            (v.readability_score or 0) / 100 * 0.4
        ))
        
        return best_variant.variant_id
    
    def _generate_cache_key(self, input_data: CopywritingInput) -> str:
        """Generate cache key for input data."""
        # Create hash of key input parameters
        key_data = {
            "product": input_data.product_description[:100],
            "platform": input_data.target_platform.value,
            "content_type": input_data.content_type.value,
            "tone": input_data.tone.value,
            "language": input_data.language.value
        }
        
        if JSON_AVAILABLE:
            key_bytes = orjson.dumps(key_data, sort_keys=True)
        else:
            key_bytes = json.dumps(key_data, sort_keys=True).encode()
        
        return f"copy:{hashlib.md5(key_bytes).hexdigest()}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[bytes]:
        """Get result from cache."""
        if not self.redis_client:
            return None
        
        try:
            result = await self.redis_client.get(cache_key)
            return result.encode() if result else None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, data: bytes):
        """Cache result with TTL."""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(
                cache_key, 
                self.settings.cache_ttl, 
                data.decode() if isinstance(data, bytes) else data
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    async def get_analytics(self, tracking_id: str) -> Dict[str, Any]:
        """Get analytics for generated content."""
        # Simulated analytics - in production, this would query a database
        return {
            "tracking_id": tracking_id,
            "total_generations": 1,
            "avg_engagement": 0.75,
            "best_performing_tone": "professional",
            "platform_performance": {
                "instagram": 0.8,
                "facebook": 0.7,
                "twitter": 0.75
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the service."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "success_rate": round((self._request_count - self._error_count) / max(self._request_count, 1) * 100, 2),
            "average_processing_time": round(self._total_processing_time / max(self._request_count, 1), 3),
            "total_processing_time": round(self._total_processing_time, 3),
            "cache_size": len(self._template_cache),
            "metrics_cache_size": len(self._metrics_cache)
        }
    
    async def cleanup(self) -> Any:
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.http_client:
            await self.http_client.aclose()
        
        self.thread_pool.shutdown(wait=True)

# New v11 Service Wrapper
class CopywritingServiceV11:
    """V11 optimized copywriting service wrapper."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.engine = None
        self._initialized = False
        self._request_count = 0
        self._error_count = 0
        self._total_processing_time = 0.0
    
    async def _ensure_engine(self):
        """Ensure the v11 engine is initialized."""
        if not self._initialized:
            self.engine = await get_engine()
            self._initialized = True
    
    async def generate(self, request: CopywritingInput) -> CopywritingOutput:
        """Generate copywriting using v11 optimized engine."""
        start_time = time.perf_counter()
        self._request_count += 1
        
        await self._ensure_engine()
        
        try:
            # Convert CopywritingInput to the format expected by v11 engine
            v11_request = {
                "product_description": request.product_description,
                "target_platform": request.target_platform.value if hasattr(request.target_platform, 'value') else str(request.target_platform),
                "tone": request.tone.value if hasattr(request.tone, 'value') else str(request.tone),
                "target_audience": request.target_audience,
                "key_points": request.key_points,
                "instructions": request.instructions,
                "restrictions": request.restrictions,
                "creativity_level": request.creativity_level,
                "language": request.language.value if hasattr(request.language, 'value') else str(request.language),
                "content_type": request.content_type.value if hasattr(request.content_type, 'value') else str(request.content_type),
                "brand_voice": request.brand_voice.value if hasattr(request.brand_voice, 'value') else str(request.brand_voice),
                "max_variants": request.max_variants,
                "tracking_id": request.tracking_id
            }
            
            result = await self.engine.generate_copywriting(v11_request)
            
            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            self._total_processing_time += processing_time
            
            # Convert v11 result back to CopywritingOutput format
            return CopywritingOutput(
                variants=result.get("variants", []),
                model_used=result.get("model_used", "v11-optimized"),
                generation_time=processing_time,
                best_variant_id=result.get("best_variant_id", ""),
                confidence_score=result.get("confidence_score", 0.85),
                tracking_id=result.get("tracking_id", request.tracking_id),
                created_at=datetime.now()
            )
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"v11 generation failed: {e}")
            raise
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from v11 engine."""
        await self._ensure_engine()
        
        engine_stats = self.engine.get_performance_stats()
        
        # Add service-level stats
        service_stats = {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "success_rate": round((self._request_count - self._error_count) / max(self._request_count, 1) * 100, 2),
            "average_processing_time": round(self._total_processing_time / max(self._request_count, 1), 3),
            "total_processing_time": round(self._total_processing_time, 3)
        }
        
        return {
            "engine_stats": engine_stats,
            "service_stats": service_stats
        }
    
    async def get_engine_info(self) -> Dict[str, Any]:
        """Get detailed engine information."""
        await self._ensure_engine()
        
        return {
            "model_name": self.model_name,
            "engine_initialized": self._initialized,
            "engine_type": "UltraOptimizedEngineV11",
            "components": {
                "intelligent_cache": hasattr(self.engine, 'intelligent_cache'),
                "memory_manager": hasattr(self.engine, 'memory_manager'),
                "batch_processor": hasattr(self.engine, 'batch_processor'),
                "circuit_breaker": hasattr(self.engine, 'circuit_breaker')
            }
        }
    
    async def optimize_config(self, config_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize engine configuration."""
        await self._ensure_engine()
        
        try:
            # Apply configuration parameters
            for key, value in config_params.items():
                if hasattr(self.engine.config, key):
                    setattr(self.engine.config, key, value)
            
            return {
                "status": "optimization_applied",
                "applied_parameters": config_params
            }
        except Exception as e:
            logger.error(f"Failed to optimize config: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup v11 engine resources."""
        if self.engine:
            await cleanup_engine()

# Legacy service for backward compatibility
class CopywritingService:
    """Legacy copywriting service for backward compatibility."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.optimized_service = OptimizedCopywritingService()
        self.v11_service = CopywritingServiceV11(model_name)
    
    async def generate(self, request: CopywritingInput) -> CopywritingOutput:
        """Generate copywriting using legacy service."""
        return await self.optimized_service.generate_copy(request)
    
    async def generate_v11(self, request: CopywritingInput) -> CopywritingOutput:
        """Generate copywriting using v11 optimized engine."""
        return await self.v11_service.generate(request)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        legacy_stats = self.optimized_service.get_performance_stats()
        v11_stats = await self.v11_service.get_performance_stats()
        
        return {
            "legacy_stats": legacy_stats,
            "v11_stats": v11_stats
        }
    
    async def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return await self.v11_service.get_engine_info()
    
    async def optimize_config(self, config_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize engine configuration."""
        return await self.v11_service.optimize_config(config_params)
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.optimized_service.cleanup()
        await self.v11_service.cleanup()

# Global service instance
_service_instance = None

async def get_copywriting_service() -> OptimizedCopywritingService:
    """Get optimized copywriting service instance."""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = OptimizedCopywritingService()
    
    return _service_instance

# Global v11 service instance
_v11_service_instance = None

async def get_copywriting_service_v11() -> CopywritingServiceV11:
    """Get v11 optimized copywriting service instance."""
    global _v11_service_instance
    
    if _v11_service_instance is None:
        _v11_service_instance = CopywritingServiceV11()
    
    return _v11_service_instance 