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
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
    import orjson
    import json as orjson
    import polars as pl
    import numpy as np
    import httpx
    import redis.asyncio as aioredis
import structlog
from prometheus_client import Counter, Histogram, Gauge
import psutil
from .models import (
            import json
        import hashlib
                    import json
                import json
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized Production Copywriting Service.

High-performance service with advanced libraries for production deployment.
Includes: orjson, polars, asyncio optimization, caching, and monitoring.
"""


# High-performance imports
try:
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

try:
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Performance monitoring

# Import models
    CopywritingInput, CopywritingOutput, CopyVariant, 
    Language, CopyTone, UseCase, CreativityLevel,
    WebsiteInfo, BrandVoice, TranslationSettings
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUESTS_TOTAL = Counter('copywriting_requests_total', 'Total copywriting requests')
REQUEST_DURATION = Histogram('copywriting_request_duration_seconds', 'Request duration')
CACHE_HITS = Counter('copywriting_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('copywriting_cache_misses_total', 'Cache misses')
ACTIVE_REQUESTS = Gauge('copywriting_active_requests', 'Active requests')

class OptimizedCopywritingService:
    """Ultra-optimized production copywriting service."""
    
    def __init__(self) -> Any:
        self.redis_client: Optional[aioredis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, mp.cpu_count() * 4))
        
        # Performance optimization
        self.template_cache = {}
        self.metrics_cache = {}
        self.generation_cache = {}
        
        # System info
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Initialize async components
        asyncio.create_task(self._initialize_async())
        
        logger.info("OptimizedCopywritingService initialized", 
                   cpu_count=self.cpu_count, 
                   memory_gb=round(self.memory_gb, 2))
    
    async def _initialize_async(self) -> Any:
        """Initialize async components."""
        try:
            # Redis for caching
            if REDIS_AVAILABLE:
                self.redis_client = await aioredis.from_url(
                    "redis://localhost:6379/0",
                    max_connections=20,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis initialized")
            
            # HTTP client for external APIs
            if HTTPX_AVAILABLE:
                self.http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    limits=httpx.Limits(max_connections=100)
                )
                logger.info("HTTP client initialized")
                
        except Exception as e:
            logger.warning("Failed to initialize async components", error=str(e))
    
    @REQUEST_DURATION.time()
    async def generate_copy(self, input_data: CopywritingInput) -> CopywritingOutput:
        """Generate optimized copywriting content."""
        REQUESTS_TOTAL.inc()
        ACTIVE_REQUESTS.inc()
        
        start_time = time.perf_counter()
        
        try:
            # Validate input
            await self._validate_input(input_data)
            
            # Check cache
            cache_key = self._generate_cache_key(input_data)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                CACHE_HITS.inc()
                ACTIVE_REQUESTS.dec()
                return cached_result
            
            CACHE_MISSES.inc()
            
            # Generate variants in parallel
            variants = await self._generate_variants_parallel(input_data)
            
            # Apply translations if requested
            if input_data.translation_settings:
                variants = await self._apply_translations(variants, input_data.translation_settings)
            
            # Calculate performance metrics
            await self._calculate_metrics_parallel(variants)
            
            # Select best variant
            best_variant_id = self._select_best_variant(variants)
            
            # Create output
            generation_time = time.perf_counter() - start_time
            output = CopywritingOutput(
                variants=variants,
                model_used="optimized-production-v2",
                generation_time=generation_time,
                best_variant_id=best_variant_id,
                confidence_score=self._calculate_confidence(variants),
                tracking_id=input_data.tracking_id,
                created_at=datetime.now(),
                performance_metrics={
                    "generation_time_ms": generation_time * 1000,
                    "variants_generated": len(variants),
                    "cache_used": False,
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
                }
            )
            
            # Cache result
            await self._cache_result(cache_key, output)
            
            logger.info("Copy generated successfully", 
                       variants=len(variants),
                       generation_time_ms=generation_time * 1000,
                       best_variant=best_variant_id)
            
            return output
            
        finally:
            ACTIVE_REQUESTS.dec()
    
    async def _validate_input(self, input_data: CopywritingInput):
        """Fast input validation."""
        if not input_data.product_description.strip():
            raise ValueError("Product description cannot be empty")
        
        if len(input_data.product_description) > 2000:
            raise ValueError("Product description too long")
    
    def _generate_cache_key(self, input_data: CopywritingInput) -> str:
        """Generate optimized cache key."""
        key_data = {
            "product": input_data.product_description[:100],
            "platform": input_data.target_platform.value,
            "tone": input_data.tone.value,
            "use_case": input_data.use_case.value,
            "language": input_data.language.value,
            "creativity": input_data.effective_creativity_score,
            "max_variants": input_data.effective_max_variants
        }
        
        if JSON_AVAILABLE:
            key_bytes = orjson.dumps(key_data, sort_keys=True)
        else:
            key_bytes = json.dumps(key_data, sort_keys=True).encode()
        
        return f"copy:v2:{hashlib.md5(key_bytes).hexdigest()}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[CopywritingOutput]:
        """Get from cache with fallback."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                if JSON_AVAILABLE:
                    data = orjson.loads(cached_data)
                else:
                    data = json.loads(cached_data)
                return CopywritingOutput(**data)
        except Exception as e:
            logger.warning("Cache get failed", error=str(e))
        
        return None
    
    async def _cache_result(self, cache_key: str, output: CopywritingOutput):
        """Cache result with TTL."""
        if not self.redis_client:
            return
        
        try:
            if JSON_AVAILABLE:
                data = orjson.dumps(output.model_dump())
            else:
                data = json.dumps(output.model_dump())
            
            await self.redis_client.setex(cache_key, 3600, data)  # 1 hour TTL
        except Exception as e:
            logger.warning("Cache set failed", error=str(e))
    
    async def _generate_variants_parallel(self, input_data: CopywritingInput) -> List[CopyVariant]:
        """Generate variants in parallel for maximum performance."""
        max_variants = input_data.effective_max_variants
        
        # Create generation tasks
        tasks = []
        for i in range(max_variants):
            task = asyncio.create_task(
                self._generate_single_variant(input_data, i)
            )
            tasks.append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        variants = [
            result for result in results 
            if isinstance(result, CopyVariant)
        ]
        
        return variants
    
    async def _generate_single_variant(self, input_data: CopywritingInput, variant_index: int) -> CopyVariant:
        """Generate a single optimized variant."""
        
        # Get optimized template
        template = self._get_template(input_data, variant_index)
        
        # Generate content components
        headline = await self._generate_headline(input_data, template, variant_index)
        primary_text = await self._generate_primary_text(input_data, template, variant_index)
        call_to_action = await self._generate_cta(input_data, variant_index)
        hashtags = await self._generate_hashtags(input_data)
        
        # Calculate basic metrics
        full_text = f"{headline} {primary_text}"
        word_count = len(full_text.split())
        char_count = len(full_text)
        
        variant = CopyVariant(
            variant_id=f"{input_data.tracking_id}_{variant_index}_{int(time.time())}",
            headline=headline,
            primary_text=primary_text,
            call_to_action=call_to_action,
            hashtags=hashtags,
            character_count=char_count,
            word_count=word_count,
            created_at=datetime.now()
        )
        
        return variant
    
    def _get_template(self, input_data: CopywritingInput, variant_index: int) -> Dict[str, str]:
        """Get optimized template based on use case and tone."""
        cache_key = f"{input_data.use_case}_{input_data.tone}_{variant_index}"
        
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
        
        # High-performance template selection
        templates = {
            UseCase.product_launch: {
                CopyTone.urgent: [
                    {"headline": "🚀 ¡{product} Ya Está Aquí!", "text": "El momento que esperabas ha llegado. {product} revoluciona {benefit}.", "cta": "¡Consíguelo Ahora!"},
                    {"headline": "⚡ Lanzamiento: {product}", "text": "Por fin disponible. {product} cambia todo lo que conocías sobre {benefit}.", "cta": "¡Pruébalo Ya!"},
                    {"headline": "🔥 {product} - Disponible HOY", "text": "No esperes más. {product} está aquí para transformar {benefit}.", "cta": "¡Obtén el Tuyo!"}
                ],
                CopyTone.professional: [
                    {"headline": "Presentamos {product}", "text": "Una nueva solución profesional para {benefit}. Diseñado para empresas que buscan excelencia.", "cta": "Solicitar Demo"},
                    {"headline": "{product} - La Nueva Era", "text": "Tecnología avanzada que redefine {benefit} para profesionales.", "cta": "Conocer Más"},
                    {"headline": "Innovación: {product}", "text": "La herramienta que los expertos estaban esperando para {benefit}.", "cta": "Explorar Funciones"}
                ]
            },
            UseCase.brand_awareness: {
                CopyTone.friendly: [
                    {"headline": "¡Hola! Somos {brand} 👋", "text": "Nos dedicamos a hacer que {benefit} sea más fácil para ti.", "cta": "¡Conócenos!"},
                    {"headline": "Te presentamos {brand}", "text": "Una marca creada pensando en ti y en cómo mejorar {benefit}.", "cta": "Descubre Más"},
                    {"headline": "{brand} está aquí", "text": "Para acompañarte en tu camino hacia {benefit}.", "cta": "¡Únete a Nosotros!"}
                ]
            }
        }
        
        use_case_templates = templates.get(input_data.use_case, {})
        tone_templates = use_case_templates.get(input_data.tone, [
            {"headline": "{product} para {benefit}", "text": "Descubre cómo {product} puede mejorar {benefit}.", "cta": "Más Información"}
        ])
        
        template = tone_templates[variant_index % len(tone_templates)]
        self.template_cache[cache_key] = template
        
        return template
    
    async def _generate_headline(self, input_data: CopywritingInput, template: Dict[str, str], variant_index: int) -> str:
        """Generate optimized headline."""
        headline_template = template.get("headline", "{product}")
        
        # Extract key info
        product_name = self._extract_product_name(input_data)
        brand_name = self._extract_brand_name(input_data)
        benefit = self._extract_benefit(input_data)
        
        # Apply creativity variations
        creativity_variations = ["", " ✨", " 🎯", " 💫", " 🌟", " ⭐", " 🔥", " 💎"]
        variation = creativity_variations[variant_index % len(creativity_variations)] if input_data.effective_creativity_score > 0.6 else ""f"
        
        headline = headline_template" + variation
        
        return headline[:200]
    
    async def _generate_primary_text(self, input_data: CopywritingInput, template: Dict[str, str], variant_index: int) -> str:
        """Generate optimized primary text."""
        text_template = template.get("text", "Descubre {product}."f")
        
        product_name = self._extract_product_name(input_data)
        benefit = self._extract_benefit(input_data)
        
        text = text_template"
        
        # Add features if available
        if input_data.website_info and input_data.website_info.features:
            features = input_data.website_info.features[:3]
            features_text = " ".join([f"✓ {feature}" for feature in features])
            text += f" {features_text}"
        
        # Add key points
        if input_data.key_points:
            key_points = input_data.key_points[:2]
            points_text = " ".join([f"• {point}" for point in key_points])
            text += f" {points_text}"
        
        return text[:2000]
    
    def _extract_product_name(self, input_data: CopywritingInput) -> str:
        """Extract product name from description."""
        if input_data.website_info and input_data.website_info.website_name:
            return input_data.website_info.website_name
        
        # Extract from description
        return input_data.product_description.split('.')[0][:50].strip()
    
    def _extract_brand_name(self, input_data: CopywritingInput) -> str:
        """Extract brand name."""
        if input_data.website_info and input_data.website_info.website_name:
            return input_data.website_info.website_name
        return "nuestra marca"
    
    def _extract_benefit(self, input_data: CopywritingInput) -> str:
        """Extract main benefit."""
        if input_data.key_points:
            return input_data.key_points[0]
        if input_data.website_info and input_data.website_info.value_proposition:
            return input_data.website_info.value_proposition[:100]
        return "tus objetivos"
    
    async def _generate_cta(self, input_data: CopywritingInput, variant_index: int) -> str:
        """Generate optimized call-to-action."""
        if input_data.call_to_action:
            return input_data.call_to_action
        
        cta_options = {
            CopyTone.urgent: ["¡Actúa Ahora!", "¡No Esperes!", "¡Solo Hoy!", "¡Aprovecha Ya!"],
            CopyTone.professional: ["Solicitar Info", "Contactar", "Ver Demo", "Consultar"],
            CopyTone.friendly: ["¡Pruébalo!", "¡Te Encantará!", "Descúbrelo", "¡Únete!"],
            CopyTone.casual: ["Échale un Vistazo", "Pruébalo", "Ver Más", "Conocer"]
        }
        
        options = cta_options.get(input_data.tone, ["Más Información", "Conocer Más", "Ver Detalles"])
        return options[variant_index % len(options)]
    
    async def _generate_hashtags(self, input_data: CopywritingInput) -> List[str]:
        """Generate optimized hashtags."""
        hashtags = []
        
        if input_data.target_platform.value in ["instagram", "twitter", "tiktok"]:
            # Extract from product description
            words = input_data.product_description.lower().split()
            relevant_words = [word for word in words if len(word) > 3][:5]
            hashtags.extend([f"#{word}" for word in relevant_words])
            
            # Add use case hashtags
            use_case_hashtags = {
                UseCase.product_launch: ["#lanzamiento", "#nuevo", "#innovacion"],
                UseCase.brand_awareness: ["#marca", "#conocenos", "#somos"],
                UseCase.social_media: ["#social", "#comunidad", "#conecta"]
            }
            hashtags.extend(use_case_hashtags.get(input_data.use_case, []))
            
            # Add website features as hashtags
            if input_data.website_info and input_data.website_info.features:
                for feature in input_data.website_info.features[:3]:
                    clean_feature = feature.replace(" ", "").replace("-", "")[:15]
                    hashtags.append(f"#{clean_feature}")
        
        return hashtags[:15]
    
    async def _apply_translations(self, variants: List[CopyVariant], translation_settings: TranslationSettings) -> List[CopyVariant]:
        """Apply translations to variants."""
        if not translation_settings.target_languages:
            return variants
        
        translated_variants = []
        
        for variant in variants:
            for language in translation_settings.target_languages:
                translated_variant = await self._translate_variant(variant, language, translation_settings)
                translated_variants.append(translated_variant)
        
        return variants + translated_variants
    
    async def _translate_variant(self, variant: CopyVariant, target_language: Language, settings: TranslationSettings) -> CopyVariant:
        """Translate a single variant (simplified implementation)."""
        # Simple translation mapping for demo
        translations = {
            Language.en: {
                "¡": "!",
                "Descubre": "Discover",
                "Pruébalo": "Try it",
                "Más Información": "Learn More"
            }
        }
        
        lang_translations = translations.get(target_language, {})
        
        # Apply basic translations
        translated_headline = variant.headline
        translated_text = variant.primary_text
        translated_cta = variant.call_to_action or ""
        
        for spanish, english in lang_translations.items():
            translated_headline = translated_headline.replace(spanish, english)
            translated_text = translated_text.replace(spanish, english)
            translated_cta = translated_cta.replace(spanish, english)
        
        return CopyVariant(
            variant_id=f"{variant.variant_id}_{target_language.value}",
            headline=translated_headline,
            primary_text=translated_text,
            call_to_action=translated_cta,
            hashtags=variant.hashtags if not settings.translate_hashtags else [],
            character_count=len(f"{translated_headline} {translated_text}"),
            word_count=len(f"{translated_headline} {translated_text}".split()),
            created_at=datetime.now()
        )
    
    async def _calculate_metrics_parallel(self, variants: List[CopyVariant]):
        """Calculate metrics for all variants in parallel."""
        tasks = [
            asyncio.create_task(self._calculate_variant_metrics(variant))
            for variant in variants
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _calculate_variant_metrics(self, variant: CopyVariant):
        """Calculate comprehensive metrics for a variant."""
        full_text = f"{variant.headline} {variant.primary_text}"
        
        # Basic metrics
        words = full_text.split()
        sentences = full_text.split('.')
        
        # Readability (simplified Flesch score)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        readability_score = max(0, min(100, 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)))
        
        # Engagement prediction (based on readability and length)
        optimal_length = 50  # words
        length_factor = 1 - abs(len(words) - optimal_length) / optimal_length
        engagement_prediction = (readability_score / 100 * 0.6) + (length_factor * 0.4)
        
        # Update variant
        variant.readability_score = readability_score
        variant.engagement_prediction = max(0, min(1, engagement_prediction))
    
    def _select_best_variant(self, variants: List[CopyVariant]) -> str:
        """Select best variant based on comprehensive scoring."""
        if not variants:
            return ""
        
        def score_variant(variant: CopyVariant) -> float:
            score = 0.0
            
            # Engagement prediction (40%)
            if variant.engagement_prediction:
                score += variant.engagement_prediction * 0.4
            
            # Readability (30%)
            if variant.readability_score:
                score += (variant.readability_score / 100) * 0.3
            
            # Length optimization (20%)
            optimal_length = 100  # characters for headline + text
            total_length = len(variant.headline) + len(variant.primary_text or "")
            length_score = 1 - abs(total_length - optimal_length) / optimal_length
            score += max(0, length_score) * 0.2
            
            # CTA presence (10%)
            if variant.call_to_action:
                score += 0.1
            
            return score
        
        best_variant = max(variants, key=score_variant)
        return best_variant.variant_id
    
    def _calculate_confidence(self, variants: List[CopyVariant]) -> float:
        """Calculate overall confidence score."""
        if not variants:
            return 0.0
        
        # Average engagement prediction
        avg_engagement = sum(
            v.engagement_prediction or 0 for v in variants
        ) / len(variants)
        
        # Variance in quality (lower variance = higher confidence)
        scores = [v.engagement_prediction or 0 for v in variants]
        if NUMPY_AVAILABLE:
            variance = np.var(scores)
            confidence = avg_engagement * (1 - min(variance, 0.5))
        else:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            confidence = avg_engagement * (1 - min(variance, 0.5))
        
        return max(0.0, min(1.0, confidence))
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        return {
            "service_info": {
                "cpu_count": self.cpu_count,
                "memory_gb": round(self.memory_gb, 2),
                "redis_available": self.redis_client is not None,
                "http_client_available": self.http_client is not None
            },
            "cache_stats": {
                "template_cache_size": len(self.template_cache),
                "metrics_cache_size": len(self.metrics_cache)
            },
            "system_stats": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "active_threads": self.thread_pool._threads
            },
            "optimization_libraries": {
                "orjson": JSON_AVAILABLE,
                "polars": POLARS_AVAILABLE,
                "numpy": NUMPY_AVAILABLE,
                "httpx": HTTPX_AVAILABLE,
                "redis": REDIS_AVAILABLE
            }
        }
    
    async def cleanup(self) -> Any:
        """Cleanup service resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.http_client:
                await self.http_client.aclose()
            
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Service cleanup completed")
            
        except Exception as e:
            logger.error("Service cleanup error", error=str(e))

# Global service instance
_service_instance: Optional[OptimizedCopywritingService] = None

async def get_optimized_service() -> OptimizedCopywritingService:
    """Get optimized service instance."""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = OptimizedCopywritingService()
    
    return _service_instance

# Export service
__all__ = ["OptimizedCopywritingService", "get_optimized_service"] 