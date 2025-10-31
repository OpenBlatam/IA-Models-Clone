from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
    import polars as pl
    import numpy as np
    import rapidfuzz
import structlog
from .config import get_config
from .cache import get_cache_manager, cached
from .optimization import get_optimization_detector
from ..models import (
from typing import Any, List, Dict, Optional
import logging
"""
Modular Copywriting Service with High-Performance Libraries.

Clean, modular service with optimized libraries:
- Parallel processing with asyncio
- Fast serialization with orjson/msgspec
- Intelligent caching
- Performance monitoring
"""


# High-performance imports
try:
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# Import models
    CopywritingInput, CopywritingOutput, CopyVariant, 
    Language, CopyTone, UseCase, CreativityLevel,
    WebsiteInfo, BrandVoice, TranslationSettings
)

logger = structlog.get_logger(__name__)

class ModularCopywritingService:
    """Modular copywriting service with high-performance optimizations."""
    
    def __init__(self) -> Any:
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self.optimization_detector = get_optimization_detector()
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(16, mp.cpu_count() * 2)
        )
        
        # Template cache
        self.template_cache = {}
        self.performance_stats = {
            "requests_processed": 0,
            "total_generation_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("ModularCopywritingService initialized",
                   optimization_level=self.optimization_detector.performance_level,
                   expected_speedup=f"{self.optimization_detector.total_speedup:.1f}x")
    
    async def initialize(self) -> Any:
        """Initialize service components."""
        await self.cache_manager.initialize()
        logger.info("ModularCopywritingService fully initialized")
    
    async def generate_copy(self, input_data: CopywritingInput) -> CopywritingOutput:
        """Generate copywriting content with optimizations."""
        start_time = time.perf_counter()
        
        try:
            # Fast input validation
            if not self._validate_input(input_data):
                raise ValueError("Invalid input data")
            
            # Check cache first
            cache_key = self._generate_cache_key(input_data)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                logger.info("Cache hit for generation", tracking_id=input_data.tracking_id)
                return CopywritingOutput(**cached_result)
            
            self.performance_stats["cache_misses"] += 1
            
            # Generate variants in parallel
            variants = await self._generate_variants_parallel(input_data)
            
            # Apply translations if requested
            if input_data.translation_settings and self.config.enable_translation:
                variants = await self._apply_translations_optimized(variants, input_data.translation_settings)
            
            # Calculate metrics with optimizations
            await self._calculate_metrics_optimized(variants)
            
            # Select best variant
            best_variant_id = self._select_best_variant_optimized(variants)
            
            # Create output
            generation_time = time.perf_counter() - start_time
            output = CopywritingOutput(
                variants=variants,
                model_used="modular-optimized-v1",
                generation_time=generation_time,
                best_variant_id=best_variant_id,
                confidence_score=self._calculate_confidence_optimized(variants),
                tracking_id=input_data.tracking_id,
                created_at=datetime.now(),
                performance_metrics={
                    "generation_time_ms": generation_time * 1000,
                    "variants_generated": len(variants),
                    "optimization_level": self.optimization_detector.performance_level,
                    "cache_used": False,
                    "optimizations": self.optimization_detector.optimizations
                }
            )
            
            # Cache result asynchronously
            asyncio.create_task(
                self.cache_manager.set(cache_key, output.model_dump())
            )
            
            # Update stats
            self.performance_stats["requests_processed"] += 1
            self.performance_stats["total_generation_time"] += generation_time
            
            logger.info("Copy generated successfully",
                       tracking_id=input_data.tracking_id,
                       variants=len(variants),
                       generation_time_ms=generation_time * 1000)
            
            return output
            
        except Exception as e:
            logger.error("Copy generation failed", 
                        tracking_id=input_data.tracking_id,
                        error=str(e))
            raise
    
    def _validate_input(self, input_data: CopywritingInput) -> bool:
        """Fast input validation."""
        return (
            input_data.product_description and
            len(input_data.product_description.strip()) > 0 and
            len(input_data.product_description) <= 2000 and
            input_data.effective_max_variants <= self.config.max_variants
        )
    
    def _generate_cache_key(self, input_data: CopywritingInput) -> str:
        """Generate optimized cache key."""
        key_components = [
            input_data.product_description[:100],
            input_data.target_platform.value,
            input_data.tone.value,
            input_data.use_case.value,
            input_data.language.value,
            str(input_data.effective_creativity_score),
            str(input_data.effective_max_variants)
        ]
        
        # Add website info if available
        if input_data.website_info:
            key_components.append(input_data.website_info.website_name or "")
        
        return "|".join(key_components)
    
    async def _generate_variants_parallel(self, input_data: CopywritingInput) -> List[CopyVariant]:
        """Generate variants with optimized parallel processing."""
        max_variants = min(input_data.effective_max_variants, self.config.max_variants)
        
        # Create tasks for parallel execution
        tasks = []
        for i in range(max_variants):
            task = asyncio.create_task(
                self._generate_single_variant_optimized(input_data, i)
            )
            tasks.append(task)
        
        # Execute all variants in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        variants = [
            result for result in results 
            if isinstance(result, CopyVariant)
        ]
        
        if not variants:
            # Fallback: create at least one basic variant
            variants = [await self._generate_basic_variant(input_data)]
        
        return variants
    
    async def _generate_single_variant_optimized(self, input_data: CopywritingInput, variant_index: int) -> CopyVariant:
        """Generate a single variant with optimizations."""
        
        # Get optimized template
        template = await self._get_template_optimized(input_data, variant_index)
        
        # Generate content components efficiently
        headline = self._generate_headline_optimized(input_data, template, variant_index)
        primary_text = self._generate_primary_text_optimized(input_data, template, variant_index)
        call_to_action = self._generate_cta_optimized(input_data, variant_index)
        hashtags = self._generate_hashtags_optimized(input_data)
        
        # Fast metrics calculation
        full_text = f"{headline} {primary_text}"
        word_count = len(full_text.split())
        char_count = len(full_text)
        
        return CopyVariant(
            variant_id=f"{input_data.tracking_id}_{variant_index}_{int(time.time() * 1000)}",
            headline=headline,
            primary_text=primary_text,
            call_to_action=call_to_action,
            hashtags=hashtags,
            character_count=char_count,
            word_count=word_count,
            created_at=datetime.now()
        )
    
    @cached(key_prefix="template", ttl=7200)  # Cache templates for 2 hours
    async def _get_template_optimized(self, input_data: CopywritingInput, variant_index: int) -> Dict[str, str]:
        """Get optimized template with caching."""
        cache_key = f"{input_data.use_case}_{input_data.tone}_{variant_index}"
        
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
        
        # Optimized template selection
        templates = self._get_template_library()
        
        use_case_templates = templates.get(input_data.use_case, {})
        tone_templates = use_case_templates.get(input_data.tone, [])
        
        if not tone_templates:
            # Fallback template
            template = {
                "headline": "{product} - {benefit}",
                "text": "Descubre {product} y mejora {benefit}.",
                "cta": "Más Información"
            }
        else:
            template = tone_templates[variant_index % len(tone_templates)]
        
        self.template_cache[cache_key] = template
        return template
    
    def _get_template_library(self) -> Dict[UseCase, Dict[CopyTone, List[Dict[str, str]]]]:
        """Get comprehensive template library."""
        return {
            UseCase.product_launch: {
                CopyTone.urgent: [
                    {
                        "headline": "🚀 ¡{product} Ya Disponible!",
                        "text": "El momento que esperabas ha llegado. {product} está aquí para revolucionar {benefit}.",
                        "cta": "¡Consíguelo Ahora!"
                    },
                    {
                        "headline": "⚡ Lanzamiento Exclusivo: {product}",
                        "text": "No esperes más. {product} cambia todo lo que conocías sobre {benefit}.",
                        "cta": "¡Pruébalo Ya!"
                    }
                ],
                CopyTone.professional: [
                    {
                        "headline": "Presentamos {product}",
                        "text": "Una nueva solución profesional diseñada para optimizar {benefit} en tu empresa.",
                        "cta": "Solicitar Demo"
                    },
                    {
                        "headline": "Innovación en {product}",
                        "text": "Tecnología avanzada que redefine los estándares de {benefit}.",
                        "cta": "Conocer Más"
                    }
                ]
            },
            UseCase.brand_awareness: {
                CopyTone.friendly: [
                    {
                        "headline": "¡Hola! Somos {brand} 👋",
                        "text": "Nos dedicamos a hacer que {benefit} sea más fácil y accesible para ti.",
                        "cta": "¡Conócenos!"
                    },
                    {
                        "headline": "Te presentamos {brand}",
                        "text": "Una marca creada con pasión, pensando en cómo mejorar {benefit}.",
                        "cta": "Descubre Más"
                    }
                ],
                CopyTone.inspirational: [
                    {
                        "headline": "✨ Transforma tu {benefit}",
                        "text": "Con {brand}, puedes alcanzar niveles increíbles de {benefit}.",
                        "cta": "¡Empieza Hoy!"
                    }
                ]
            },
            UseCase.social_media: {
                CopyTone.casual: [
                    {
                        "headline": "¿Ya conoces {product}? 🤔",
                        "text": "Te va a encantar. Es perfecto para mejorar {benefit} de forma súper fácil.",
                        "cta": "¡Échale un vistazo!"
                    }
                ]
            }
        }
    
    def _generate_headline_optimized(self, input_data: CopywritingInput, template: Dict[str, str], variant_index: int) -> str:
        """Generate optimized headline with smart replacements."""
        headline_template = template.get("headline", "{product}")
        
        # Extract information efficiently
        product_name = self._extract_product_name(input_data)
        brand_name = self._extract_brand_name(input_data)
        benefit = self._extract_primary_benefit(input_data)
        
        # Apply creativity variations
        if input_data.effective_creativity_score > 0.6:
            creativity_emojis = ["✨", "🌟", "💫", "🔥", "⚡", "🎯", "💎", "🚀"]
            emoji = creativity_emojis[variant_index % len(creativity_emojis)]
            if not any(e in headline_template for e in creativity_emojis):
                headline_template = f"{emoji} {headline_template}"f"
        
        # Smart replacements
        headline = headline_template"
        
        return headline[:200]  # Ensure reasonable length
    
    def _generate_primary_text_optimized(self, input_data: CopywritingInput, template: Dict[str, str], variant_index: int) -> str:
        """Generate optimized primary text."""
        text_template = template.get("text", "Descubre {product}."f")
        
        product_name = self._extract_product_name(input_data)
        benefit = self._extract_primary_benefit(input_data)
        
        # Base text
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
        
        return text[:1500]  # Platform-appropriate length
    
    def _extract_product_name(self, input_data: CopywritingInput) -> str:
        """Extract product name efficiently."""
        if input_data.website_info and input_data.website_info.website_name:
            return input_data.website_info.website_name
        
        # Extract from description (first sentence, first 50 chars)
        return input_data.product_description.split('.')[0][:50].strip()
    
    def _extract_brand_name(self, input_data: CopywritingInput) -> str:
        """Extract brand name efficiently."""
        if input_data.website_info and input_data.website_info.website_name:
            return input_data.website_info.website_name
        return "nuestra marca"
    
    def _extract_primary_benefit(self, input_data: CopywritingInput) -> str:
        """Extract primary benefit efficiently."""
        if input_data.key_points:
            return input_data.key_points[0][:50]
        
        if input_data.website_info and input_data.website_info.value_proposition:
            return input_data.website_info.value_proposition[:50]
        
        return "tus objetivos"
    
    def _generate_cta_optimized(self, input_data: CopywritingInput, variant_index: int) -> str:
        """Generate optimized call-to-action."""
        if input_data.call_to_action:
            return input_data.call_to_action
        
        # Optimized CTA matrix
        cta_options = {
            CopyTone.urgent: ["¡Actúa Ahora!", "¡No Esperes!", "¡Aprovecha Ya!", "¡Solo Hoy!"],
            CopyTone.professional: ["Solicitar Información", "Contactar Equipo", "Ver Demo", "Consultar"],
            CopyTone.friendly: ["¡Pruébalo!", "¡Te Encantará!", "Descúbrelo", "¡Únete!"],
            CopyTone.casual: ["Échale un Vistazo", "Ver Más", "Probar", "Conocer"],
            CopyTone.inspirational: ["¡Empieza Hoy!", "¡Transforma Ya!", "¡Logra Más!", "¡Alcanza Tus Metas!"]
        }
        
        options = cta_options.get(input_data.tone, ["Más Información", "Conocer Más"])
        return options[variant_index % len(options)]
    
    def _generate_hashtags_optimized(self, input_data: CopywritingInput) -> List[str]:
        """Generate optimized hashtags for social platforms."""
        if input_data.target_platform.value not in ["instagram", "twitter", "tiktok"]:
            return []
        
        hashtags = []
        
        # Extract keywords from product description
        if RAPIDFUZZ_AVAILABLE:
            # Use rapidfuzz for smart keyword extraction
            words = input_data.product_description.lower().split()
            relevant_words = [word for word in words if len(word) > 3 and word.isalpha()][:5]
        else:
            # Fallback to simple extraction
            words = input_data.product_description.lower().split()
            relevant_words = [word for word in words if len(word) > 3][:5]
        
        hashtags.extend([f"#{word}" for word in relevant_words])
        
        # Add use case specific hashtags
        use_case_hashtags = {
            UseCase.product_launch: ["#lanzamiento", "#nuevo", "#innovacion"],
            UseCase.brand_awareness: ["#marca", "#conocenos", "#somos"],
            UseCase.social_media: ["#social", "#comunidad", "#conecta"]
        }
        
        hashtags.extend(use_case_hashtags.get(input_data.use_case, []))
        
        # Add website features as hashtags
        if input_data.website_info and input_data.website_info.features:
            for feature in input_data.website_info.features[:3]:
                clean_feature = ''.join(c for c in feature if c.isalnum())[:15]
                if clean_feature:
                    hashtags.append(f"#{clean_feature}")
        
        return hashtags[:12]  # Reasonable limit
    
    async def _apply_translations_optimized(self, variants: List[CopyVariant], settings: TranslationSettings) -> List[CopyVariant]:
        """Apply translations with optimized processing."""
        if not settings.target_languages:
            return variants
        
        translated_variants = []
        
        # Process translations in parallel
        translation_tasks = []
        for variant in variants:
            for language in settings.target_languages:
                task = asyncio.create_task(
                    self._translate_variant_optimized(variant, language, settings)
                )
                translation_tasks.append(task)
        
        translated_results = await asyncio.gather(*translation_tasks, return_exceptions=True)
        
        # Filter successful translations
        for result in translated_results:
            if isinstance(result, CopyVariant):
                translated_variants.append(result)
        
        return variants + translated_variants
    
    async def _translate_variant_optimized(self, variant: CopyVariant, target_language: Language, settings: TranslationSettings) -> CopyVariant:
        """Translate variant with optimized string operations."""
        # Optimized translation mappings
        translation_maps = {
            Language.en: {
                "Descubre": "Discover",
                "Pruébalo": "Try it",
                "¡": "!",
                "Más Información": "Learn More",
                "Solicitar": "Request",
                "Contactar": "Contact",
                "Ver Demo": "View Demo",
                "Conocer": "Learn",
                "Empieza": "Start",
                "Transforma": "Transform"
            },
            Language.fr: {
                "Descubre": "Découvrez",
                "Pruébalo": "Essayez-le",
                "Más Información": "Plus d'infos"
            }
        }
        
        translations = translation_maps.get(target_language, {})
        
        # Optimized string replacement
        translated_headline = variant.headline
        translated_text = variant.primary_text
        translated_cta = variant.call_to_action or ""
        
        for spanish, translated in translations.items():
            translated_headline = translated_headline.replace(spanish, translated)
            translated_text = translated_text.replace(spanish, translated)
            translated_cta = translated_cta.replace(spanish, translated)
        
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
    
    async def _calculate_metrics_optimized(self, variants: List[CopyVariant]):
        """Calculate metrics with optimized algorithms."""
        if NUMPY_AVAILABLE:
            await self._calculate_metrics_numpy(variants)
        else:
            await self._calculate_metrics_standard(variants)
    
    async def _calculate_metrics_numpy(self, variants: List[CopyVariant]):
        """Calculate metrics using NumPy for optimization."""
        for variant in variants:
            full_text = f"{variant.headline} {variant.primary_text}"
            words = full_text.split()
            sentences = full_text.split('.')
            
            # Vectorized calculations
            word_lengths = np.array([len(word) for word in words])
            word_count = len(words)
            sentence_count = max(len(sentences), 1)
            
            avg_word_length = np.mean(word_lengths) if len(word_lengths) > 0 else 0
            avg_sentence_length = word_count / sentence_count
            
            # Optimized readability calculation
            readability = max(0, min(100, 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)))
            
            # Engagement prediction
            optimal_length = 50
            length_factor = 1 - abs(word_count - optimal_length) / optimal_length
            engagement = max(0, min(1, (readability / 100 * 0.6) + (length_factor * 0.4)))
            
            variant.readability_score = readability
            variant.engagement_prediction = engagement
    
    async def _calculate_metrics_standard(self, variants: List[CopyVariant]):
        """Calculate metrics using standard Python."""
        for variant in variants:
            full_text = f"{variant.headline} {variant.primary_text}"
            words = full_text.split()
            sentences = full_text.split('.')
            
            word_count = len(words)
            sentence_count = max(len(sentences), 1)
            avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
            avg_sentence_length = word_count / sentence_count
            
            # Readability calculation
            readability = max(0, min(100, 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)))
            
            # Engagement prediction
            optimal_length = 50
            length_factor = 1 - abs(word_count - optimal_length) / optimal_length
            engagement = max(0, min(1, (readability / 100 * 0.6) + (length_factor * 0.4)))
            
            variant.readability_score = readability
            variant.engagement_prediction = engagement
    
    def _select_best_variant_optimized(self, variants: List[CopyVariant]) -> str:
        """Select best variant with optimized scoring."""
        if not variants:
            return ""
        
        if NUMPY_AVAILABLE:
            # Vectorized scoring
            engagement_scores = np.array([v.engagement_prediction or 0 for v in variants])
            readability_scores = np.array([v.readability_score or 0 for v in variants])
            
            # Combined scoring
            combined_scores = (engagement_scores * 0.6) + (readability_scores / 100 * 0.4)
            best_index = np.argmax(combined_scores)
            
            return variants[best_index].variant_id
        else:
            # Standard scoring
            def score_variant(variant: CopyVariant) -> float:
                engagement = variant.engagement_prediction or 0
                readability = (variant.readability_score or 0) / 100
                return (engagement * 0.6) + (readability * 0.4)
            
            best_variant = max(variants, key=score_variant)
            return best_variant.variant_id
    
    def _calculate_confidence_optimized(self, variants: List[CopyVariant]) -> float:
        """Calculate confidence with optimized algorithms."""
        if not variants:
            return 0.0
        
        if NUMPY_AVAILABLE:
            scores = np.array([v.engagement_prediction or 0 for v in variants])
            avg_score = np.mean(scores)
            variance = np.var(scores)
            confidence = avg_score * (1 - min(variance, 0.5))
            return float(max(0.0, min(1.0, confidence)))
        else:
            scores = [v.engagement_prediction or 0 for v in variants]
            avg_score = sum(scores) / len(scores)
            return max(0.0, min(1.0, avg_score))
    
    async def _generate_basic_variant(self, input_data: CopywritingInput) -> CopyVariant:
        """Generate basic variant as fallback."""
        product_name = self._extract_product_name(input_data)
        
        return CopyVariant(
            variant_id=f"{input_data.tracking_id}_fallback_{int(time.time())}",
            headline=f"Descubre {product_name}",
            primary_text=f"La mejor solución para {input_data.target_audience or 'ti'}. {input_data.product_description[:100]}",
            call_to_action="Más Información",
            character_count=100,
            word_count=15,
            created_at=datetime.now()
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        cache_stats = self.cache_manager.get_stats()
        
        avg_generation_time = 0.0
        if self.performance_stats["requests_processed"] > 0:
            avg_generation_time = (
                self.performance_stats["total_generation_time"] / 
                self.performance_stats["requests_processed"]
            )
        
        return {
            "service_stats": self.performance_stats,
            "avg_generation_time_ms": avg_generation_time * 1000,
            "cache_stats": cache_stats,
            "optimization_level": self.optimization_detector.performance_level,
            "expected_speedup": f"{self.optimization_detector.total_speedup:.1f}x",
            "optimizations": self.optimization_detector.optimizations
        }
    
    async def cleanup(self) -> Any:
        """Cleanup service resources."""
        try:
            await self.cache_manager.cleanup()
            self.thread_pool.shutdown(wait=True)
            logger.info("ModularCopywritingService cleanup completed")
        except Exception as e:
            logger.error("Service cleanup error", error=str(e))

# Global service instance
_service: Optional[ModularCopywritingService] = None

async def get_service() -> ModularCopywritingService:
    """Get modular service instance."""
    global _service
    if _service is None:
        _service = ModularCopywritingService()
        await _service.initialize()
    return _service

# Export service
__all__ = ["ModularCopywritingService", "get_service"] 