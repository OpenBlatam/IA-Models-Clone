from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import uuid
import json
from contextlib import asynccontextmanager
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import AsyncCallbackHandler
from langchain.llms.base import LLM
import httpx
from .config import get_config, CopywritingConfig
from .models import (
from .optimization import get_optimization_manager, OptimizationManager
from .cache import get_cache_manager, CacheManager
from .monitoring import get_metrics_collector, MetricsCollector
                from langchain.chat_models import ChatAnthropic
                from langchain.chat_models import ChatGooglePalm
from typing import Any, List, Dict, Optional
"""
Copywriting Service
==================

High-performance, production-ready copywriting service with LangChain integration,
multi-AI support, advanced caching, and comprehensive optimization.
"""


# LangChain imports

# OpenRouter integration

    CopywritingRequest, CopywritingResponse, ContentVariant, TranslatedContent,
    GenerationMetrics, BatchCopywritingRequest, BatchCopywritingResponse,
    LanguageEnum, ToneEnum, UseCaseEnum, AIProviderEnum
)

# Configure logging
logger = logging.getLogger(__name__)


class OpenRouterLLM(LLM):
    """Custom LangChain LLM for OpenRouter integration"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4", base_url: str = "https://openrouter.ai/api/v1"):
        
    """__init__ function."""
super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://blatam-academy.com",
                "X-Title": "Blatam Academy Copywriting"
            },
            timeout=30.0
        )
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Async call to OpenRouter API"""
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "stop": stop
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Sync call (not recommended for production)"""
        return asyncio.run(self._acall(prompt, stop))


class CopywritingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for monitoring LangChain operations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        
    """__init__ function."""
self.metrics_collector = metrics_collector
        self.start_time = None
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts"""
        self.start_time = time.time()
    
    async def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends"""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.record_histogram("langchain_llm_duration", duration)
    
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM errors"""
        self.metrics_collector.record_counter("langchain_llm_errors")


class CopywritingService:
    """Main copywriting service with advanced features"""
    
    def __init__(self, config: Optional[CopywritingConfig] = None):
        
    """__init__ function."""
self.config = config or get_config()
        self.optimization_manager = get_optimization_manager()
        self.cache_manager = get_cache_manager()
        self.metrics_collector = get_metrics_collector()
        
        # Initialize AI providers
        self.ai_providers: Dict[str, Any] = {}
        self.callback_handler = CopywritingCallbackHandler(self.metrics_collector)
        
        # Performance optimizations
        self.serializer = self.optimization_manager.get_serializer()
        self.hasher = self.optimization_manager.get_hasher()
        
        # Initialize service
        self._setup_ai_providers()
        self._setup_optimization()
        
        logger.info("✓ CopywritingService initialized with optimizations")
    
    def _setup_ai_providers(self) -> Any:
        """Setup AI providers based on configuration"""
        # OpenRouter
        if self.config.ai.openrouter_api_key:
            self.ai_providers["openrouter"] = OpenRouterLLM(
                api_key=self.config.ai.openrouter_api_key,
                base_url=self.config.ai.openrouter_base_url
            )
            logger.info("✓ OpenRouter provider initialized")
        
        # OpenAI
        if self.config.ai.openai_api_key:
            self.ai_providers["openai"] = ChatOpenAI(
                openai_api_key=self.config.ai.openai_api_key,
                model_name=self.config.ai.default_model,
                temperature=self.config.ai.temperature,
                max_tokens=self.config.ai.max_tokens,
                callbacks=[self.callback_handler]
            )
            logger.info("✓ OpenAI provider initialized")
        
        # Anthropic
        if self.config.ai.anthropic_api_key:
            try:
                self.ai_providers["anthropic"] = ChatAnthropic(
                    anthropic_api_key=self.config.ai.anthropic_api_key,
                    model="claude-3-opus-20240229",
                    callbacks=[self.callback_handler]
                )
                logger.info("✓ Anthropic provider initialized")
            except ImportError:
                logger.warning("Anthropic provider not available (missing langchain-anthropic)")
        
        # Google
        if self.config.ai.google_api_key:
            try:
                self.ai_providers["google"] = ChatGooglePalm(
                    google_api_key=self.config.ai.google_api_key,
                    callbacks=[self.callback_handler]
                )
                logger.info("✓ Google provider initialized")
            except ImportError:
                logger.warning("Google provider not available (missing langchain-google-genai)")
        
        if not self.ai_providers:
            raise ValueError("No AI providers configured. Please set at least one API key.")
    
    def _setup_optimization(self) -> Any:
        """Setup performance optimizations"""
        # Setup event loop optimization
        self.optimization_manager.setup_event_loop()
        
        # JIT compile critical functions
        self._generate_prompt = self.optimization_manager.jit_compile(
            self._generate_prompt_impl, "generate_prompt"
        )
    
    def _get_ai_provider(self, provider_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get AI provider with fallback logic"""
        if provider_name and provider_name in self.ai_providers:
            return self.ai_providers[provider_name]
        
        # Fallback to first available provider
        if self.ai_providers:
            return next(iter(self.ai_providers.values()))
        
        raise ValueError("No AI providers available")
    
    def _generate_system_prompt(self, request: CopywritingRequest) -> str:
        """Generate system prompt based on request parameters"""
        prompt_parts = [
            f"You are an expert copywriter specializing in {request.use_case.value} content.",
            f"Write in {request.language.value} language with a {request.tone.value} tone."
        ]
        
        if request.target_audience:
            prompt_parts.append(f"Target audience: {request.target_audience}")
        
        if request.keywords:
            prompt_parts.append(f"Include these keywords naturally: {', '.join(request.keywords)}")
        
        if request.website_info:
            website_context = []
            if request.website_info.name:
                website_context.append(f"Company: {request.website_info.name}")
            if request.website_info.description:
                website_context.append(f"Description: {request.website_info.description}")
            if request.website_info.value_proposition:
                website_context.append(f"Value proposition: {request.website_info.value_proposition}")
            
            if website_context:
                prompt_parts.append("Company context: " + " | ".join(website_context))
        
        if request.brand_voice:
            brand_context = []
            if request.brand_voice.personality_traits:
                brand_context.append(f"Brand personality: {', '.join(request.brand_voice.personality_traits)}")
            if request.brand_voice.communication_style:
                brand_context.append(f"Communication style: {request.brand_voice.communication_style}")
            if request.brand_voice.values:
                brand_context.append(f"Brand values: {', '.join(request.brand_voice.values)}")
            
            if brand_context:
                prompt_parts.append("Brand voice: " + " | ".join(brand_context))
        
        # Length guidance
        length_guidance = {
            "short": "Keep it concise and impactful (50-150 words).",
            "medium": "Provide comprehensive content (150-400 words).",
            "long": "Create detailed, in-depth content (400+ words)."
        }
        prompt_parts.append(length_guidance.get(request.length, length_guidance["medium"]))
        
        # Quality guidelines
        prompt_parts.extend([
            "Ensure the content is:",
            "- Engaging and compelling",
            "- Clear and easy to understand",
            "- Action-oriented when appropriate",
            "- Grammatically correct and well-structured",
            "- Optimized for the intended use case"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_prompt_impl(self, request: CopywritingRequest) -> str:
        """Generate the main prompt for content generation"""
        return f"""
{self._generate_system_prompt(request)}

User Request: {request.prompt}

Please generate high-quality {request.use_case.value} content that meets all the specified requirements.
"""
    
    async def _call_ai_provider(self, provider: Any, prompt: str, request: CopywritingRequest) -> str:
        """Call AI provider with error handling and retries"""
        max_retries = self.config.ai.max_retries
        timeout = self.config.ai.timeout
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                if isinstance(provider, OpenRouterLLM):
                    result = await provider._acall(prompt)
                else:
                    # LangChain provider
                    messages = [
                        SystemMessage(content=self._generate_system_prompt(request)),
                        HumanMessage(content=request.prompt)
                    ]
                    response = await provider.agenerate([messages])
                    result = response.generations[0][0].text
                
                duration = time.time() - start_time
                
                # Record metrics
                provider_name = getattr(provider, '_llm_type', 'unknown')
                model_name = getattr(provider, 'model_name', getattr(provider, 'model', 'unknown'))
                
                self.metrics_collector.record_ai_request(
                    provider=provider_name,
                    model=model_name,
                    tokens=len(result.split()),  # Rough token estimate
                    duration=duration,
                    success=True
                )
                
                return result.strip()
                
            except Exception as e:
                logger.warning(f"AI provider call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    # Record failed metrics
                    provider_name = getattr(provider, '_llm_type', 'unknown')
                    model_name = getattr(provider, 'model_name', getattr(provider, 'model', 'unknown'))
                    
                    self.metrics_collector.record_ai_request(
                        provider=provider_name,
                        model=model_name,
                        tokens=0,
                        duration=0,
                        success=False
                    )
                    raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
    
    async def _generate_variants(self, request: CopywritingRequest, primary_content: str) -> List[ContentVariant]:
        """Generate content variants"""
        if request.variant_settings.count <= 1:
            return []
        
        variants = []
        variant_count = min(request.variant_settings.count - 1, 5)  # Max 5 additional variants
        
        for i in range(variant_count):
            try:
                # Modify request for variant
                variant_request = request.copy(deep=True)
                
                if request.variant_settings.tone_variations and i < len(self.config.supported_tones):
                    # Use different tones for variants
                    available_tones = [t for t in self.config.supported_tones if t != request.tone.value]
                    if available_tones:
                        variant_request.tone = ToneEnum(available_tones[i % len(available_tones)])
                
                if request.variant_settings.length_variations:
                    # Vary length
                    lengths = ["short", "medium", "long"]
                    current_length = request.length or "medium"
                    other_lengths = [l for l in lengths if l != current_length]
                    if other_lengths:
                        variant_request.length = other_lengths[i % len(other_lengths)]
                
                # Generate variant
                provider = self._get_ai_provider(request.ai_provider)
                prompt = self._generate_prompt(variant_request)
                content = await self._call_ai_provider(provider, prompt, variant_request)
                
                variant = ContentVariant(
                    content=content,
                    tone=variant_request.tone,
                    length=variant_request.length or "medium",
                    word_count=len(content.split()),
                    character_count=len(content)
                )
                variants.append(variant)
                
            except Exception as e:
                logger.warning(f"Failed to generate variant {i + 1}: {e}")
                continue
        
        return variants
    
    async def _translate_content(self, content: str, request: CopywritingRequest) -> List[TranslatedContent]:
        """Translate content to target languages"""
        if not request.translation_settings or not request.translation_settings.target_languages:
            return []
        
        translations = []
        
        for target_language in request.translation_settings.target_languages:
            try:
                # Create translation request
                translation_prompt = f"""
Translate the following {request.language.value} text to {target_language.value}.
Maintain the original tone, style, and intent.
{"Apply cultural adaptation for the target market." if request.translation_settings.cultural_adaptation else ""}

Original text:
{content}

Translated text:
"""
                
                provider = self._get_ai_provider(request.ai_provider)
                translated_content = await self._call_ai_provider(provider, translation_prompt, request)
                
                translation = TranslatedContent(
                    language=target_language,
                    content=translated_content,
                    cultural_notes="Cultural adaptation applied" if request.translation_settings.cultural_adaptation else None,
                    confidence_score=0.9  # Placeholder - could be enhanced with actual scoring
                )
                translations.append(translation)
                
            except Exception as e:
                logger.warning(f"Failed to translate to {target_language.value}: {e}")
                continue
        
        return translations
    
    @asynccontextmanager
    async def _performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics_collector.record_histogram(f"{operation_name}_duration", duration)
    
    async def generate_copy(self, request: CopywritingRequest) -> CopywritingResponse:
        """Generate copywriting content"""
        request_start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self.hasher(self.serializer["dumps"](request.dict()))
            
            # Check cache first
            async with self._performance_monitoring("cache_lookup"):
                cached_result = await self.cache_manager.get(f"copy:{cache_key}")
                if cached_result:
                    self.metrics_collector.record_cache_operation("get", True, 0)
                    return CopywritingResponse(**cached_result)
                
                self.metrics_collector.record_cache_operation("get", False, 0)
            
            # Generate primary content
            async with self._performance_monitoring("ai_generation"):
                provider = self._get_ai_provider(request.ai_provider)
                prompt = self._generate_prompt(request)
                primary_content = await self._call_ai_provider(provider, prompt, request)
            
            # Generate variants if requested
            variants = []
            if request.variant_settings.count > 1:
                async with self._performance_monitoring("variant_generation"):
                    variants = await self._generate_variants(request, primary_content)
            
            # Generate translations if requested
            translations = []
            if request.translation_settings:
                async with self._performance_monitoring("translation"):
                    translations = await self._translate_content(primary_content, request)
            
            # Calculate metrics
            generation_time = time.time() - request_start_time
            provider_name = getattr(provider, '_llm_type', 'unknown')
            model_name = getattr(provider, 'model_name', getattr(provider, 'model', 'unknown'))
            
            metrics = GenerationMetrics(
                generation_time=generation_time,
                token_count=len(primary_content.split()),
                cache_hit=False,
                ai_provider=provider_name,
                model_used=model_name,
                optimization_score=self.optimization_manager.profile.performance_multiplier
            )
            
            # Create response
            response = CopywritingResponse(
                primary_content=primary_content,
                variants=variants,
                translations=translations,
                metrics=metrics,
                keywords_used=request.keywords or [],
                seo_score=self._calculate_seo_score(primary_content, request.keywords or [])
            )
            
            # Cache the result
            async with self._performance_monitoring("cache_store"):
                await self.cache_manager.set(
                    f"copy:{cache_key}",
                    response.dict(),
                    ttl=self.config.cache.redis_cache_ttl
                )
                self.metrics_collector.record_cache_operation("set", False, 0)
            
            # Record request metrics
            self.metrics_collector.record_counter("copywriting_requests_total")
            self.metrics_collector.record_histogram("copywriting_request_duration", generation_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Copy generation failed: {e}")
            self.metrics_collector.record_counter("copywriting_requests_error")
            raise
    
    def _calculate_seo_score(self, content: str, keywords: List[str]) -> float:
        """Calculate basic SEO score"""
        if not keywords:
            return 0.0
        
        content_lower = content.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
        
        # Basic scoring: percentage of keywords found
        base_score = (keyword_matches / len(keywords)) * 100
        
        # Bonus for content length
        word_count = len(content.split())
        length_bonus = min(word_count / 300 * 10, 10)  # Up to 10 points for 300+ words
        
        return min(base_score + length_bonus, 100.0)
    
    async def generate_batch(self, batch_request: BatchCopywritingRequest) -> BatchCopywritingResponse:
        """Generate multiple copywriting pieces"""
        batch_start_time = time.time()
        results = []
        successful_requests = 0
        failed_requests = 0
        
        if batch_request.parallel_processing:
            # Process requests in parallel
            tasks = []
            for request in batch_request.requests:
                task = asyncio.create_task(self._safe_generate_copy(request))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process requests sequentially
            for request in batch_request.requests:
                try:
                    result = await self.generate_copy(request)
                    results.append(result)
                    successful_requests += 1
                except Exception as e:
                    results.append(str(e))
                    failed_requests += 1
                    if batch_request.fail_fast:
                        break
        
        # Count results
        for result in results:
            if isinstance(result, CopywritingResponse):
                successful_requests += 1
            else:
                failed_requests += 1
        
        batch_metrics = {
            "total_time": time.time() - batch_start_time,
            "average_time_per_request": (time.time() - batch_start_time) / len(batch_request.requests),
            "parallel_processing": batch_request.parallel_processing
        }
        
        return BatchCopywritingResponse(
            total_requests=len(batch_request.requests),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            results=results,
            batch_metrics=batch_metrics,
            completed_at=datetime.utcnow()
        )
    
    async def _safe_generate_copy(self, request: CopywritingRequest) -> Union[CopywritingResponse, str]:
        """Safely generate copy with error handling"""
        try:
            return await self.generate_copy(request)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": self.metrics_collector.get_uptime(),
            "version": "2.0.0"
        }
        
        # Check AI providers
        ai_status = {}
        for name, provider in self.ai_providers.items():
            try:
                # Simple test call
                test_prompt = "Hello"
                if isinstance(provider, OpenRouterLLM):
                    await provider._acall(test_prompt)
                else:
                    messages = [HumanMessage(content=test_prompt)]
                    await provider.agenerate([messages])
                ai_status[name] = "healthy"
            except Exception as e:
                ai_status[name] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
        
        health["ai_providers_status"] = ai_status
        
        # Check cache health
        cache_health = await self.cache_manager.health_check()
        health["cache_status"] = cache_health
        
        # Add performance metrics
        health["performance"] = self.optimization_manager.get_performance_stats()
        health["cache_stats"] = self.cache_manager.get_stats()
        
        return health
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return {
            "service_metrics": self.metrics_collector.get_metrics_summary(),
            "optimization_metrics": self.optimization_manager.get_performance_stats(),
            "cache_metrics": self.cache_manager.get_stats()
        }
    
    async def cleanup(self) -> Any:
        """Cleanup service resources"""
        try:
            # Close AI provider connections
            for provider in self.ai_providers.values():
                if hasattr(provider, 'client') and hasattr(provider.client, 'aclose'):
                    await provider.client.aclose()
            
            # Cleanup optimization manager
            self.optimization_manager.cleanup()
            
            logger.info("✓ CopywritingService cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during service cleanup: {e}")


# Global service instance
_service_instance: Optional[CopywritingService] = None


async def get_copywriting_service() -> CopywritingService:
    """Get or create the global copywriting service instance"""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = CopywritingService()
    
    return _service_instance


async def cleanup_service():
    """Cleanup the global service instance"""
    global _service_instance
    
    if _service_instance:
        await _service_instance.cleanup()
        _service_instance = None 