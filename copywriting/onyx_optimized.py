from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
    import orjson
    import json as JSON_LIB
    import uvloop
    import redis.asyncio as aioredis
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain
    from langchain.callbacks import AsyncCallbackHandler
    import openai
    from prometheus_fastapi_instrumentator import Instrumentator
import structlog
from .models import CopywritingInput, CopywritingOutput, CopyVariant, Language, CopyTone, UseCase
        import hashlib
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Onyx-Optimized Copywriting Service.

Production-ready copywriting service integrated with Onyx backend:
- LangChain integration for AI orchestration
- OpenRouter support for multiple AI APIs
- Onyx backend compatibility
- Ultra-optimized performance
- Production monitoring and caching
"""


# FastAPI and core dependencies

# Optimization libraries with fallbacks
try:
    JSON_LIB = orjson
    JSON_SPEEDUP = 5.0
except ImportError:
    JSON_SPEEDUP = 1.0

try:
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# LangChain imports
try:
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# OpenRouter integration
try:
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Monitoring
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Import models

# === ONYX CONFIGURATION ===
class OnyxConfig:
    """Onyx-specific configuration."""
    
    def __init__(self) -> Any:
        # Onyx backend settings
        self.onyx_api_key = os.getenv("ONYX_API_KEY", "onyx-secret-key")
        self.onyx_base_url = os.getenv("ONYX_BASE_URL", "http://localhost:8080")
        
        # AI API settings
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Default models
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        self.fallback_model = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
        
        # Performance settings
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.timeout = int(os.getenv("AI_TIMEOUT", 30))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 1000))
        self.temperature = float(os.getenv("TEMPERATURE", 0.7))
        
        # Cache settings
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", 3600))
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/9")
        
        # Performance optimization
        self.performance_level = self._calculate_performance_level()
    
    def _calculate_performance_level(self) -> str:
        """Calculate performance level based on available libraries."""
        score = 0
        if JSON_SPEEDUP > 1.0:
            score += 1
        if UVLOOP_AVAILABLE:
            score += 1
        if REDIS_AVAILABLE:
            score += 1
        if LANGCHAIN_AVAILABLE:
            score += 1
        if OPENAI_AVAILABLE:
            score += 1
        
        if score >= 4:
            return "ULTRA"
        elif score >= 3:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        else:
            return "BASIC"

config = OnyxConfig()

# === AI PROVIDER MANAGER ===
class AIProviderManager:
    """Manage multiple AI providers with OpenRouter and LangChain."""
    
    def __init__(self) -> Any:
        self.providers = {}
        self.current_provider = None
        self.fallback_providers = []
        self._initialize_providers()
    
    def _initialize_providers(self) -> Any:
        """Initialize available AI providers."""
        
        # OpenRouter provider
        if config.openrouter_api_key:
            self.providers["openrouter"] = {
                "type": "openrouter",
                "api_key": config.openrouter_api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "models": [
                    "anthropic/claude-3-sonnet",
                    "openai/gpt-4-turbo",
                    "openai/gpt-3.5-turbo",
                    "google/gemini-pro",
                    "meta-llama/llama-2-70b-chat",
                    "mistralai/mixtral-8x7b-instruct"
                ]
            }
        
        # OpenAI provider
        if config.openai_api_key:
            self.providers["openai"] = {
                "type": "openai",
                "api_key": config.openai_api_key,
                "base_url": "https://api.openai.com/v1",
                "models": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            }
        
        # Anthropic provider
        if config.anthropic_api_key:
            self.providers["anthropic"] = {
                "type": "anthropic",
                "api_key": config.anthropic_api_key,
                "base_url": "https://api.anthropic.com",
                "models": ["claude-3-sonnet", "claude-3-haiku", "claude-2"]
            }
        
        # Set default provider
        if "openrouter" in self.providers:
            self.current_provider = "openrouter"
        elif "openai" in self.providers:
            self.current_provider = "openai"
        elif "anthropic" in self.providers:
            self.current_provider = "anthropic"
        
        # Set fallback order
        self.fallback_providers = [p for p in self.providers.keys() if p != self.current_provider]
        
        logger.info("AI providers initialized", 
                   providers=list(self.providers.keys()),
                   current=self.current_provider)
    
    async def get_ai_client(self, provider: Optional[str] = None):
        """Get AI client for specified provider."""
        provider = provider or self.current_provider
        
        if not provider or provider not in self.providers:
            raise HTTPException(status_code=500, detail="No AI provider available")
        
        provider_config = self.providers[provider]
        
        if LANGCHAIN_AVAILABLE:
            if provider_config["type"] == "openrouter":
                return ChatOpenAI(
                    openai_api_key=provider_config["api_key"],
                    openai_api_base=provider_config["base_url"],
                    model_name=config.default_model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout
                )
            elif provider_config["type"] == "openai":
                return ChatOpenAI(
                    openai_api_key=provider_config["api_key"],
                    model_name=config.default_model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout
                )
        
        # Fallback to direct API calls
        if OPENAI_AVAILABLE:
            openai.api_key = provider_config["api_key"]
            if provider_config["type"] == "openrouter":
                openai.api_base = provider_config["base_url"]
            return openai
        
        raise HTTPException(status_code=500, detail="No compatible AI client available")
    
    async def call_ai_with_fallback(self, prompt: str, **kwargs) -> str:
        """Call AI with automatic fallback to other providers."""
        providers_to_try = [self.current_provider] + self.fallback_providers
        
        for provider in providers_to_try:
            try:
                client = await self.get_ai_client(provider)
                
                if LANGCHAIN_AVAILABLE and hasattr(client, 'apredict'):
                    response = await client.apredict(prompt)
                    return response
                elif OPENAI_AVAILABLE and hasattr(client, 'ChatCompletion'):
                    response = await client.ChatCompletion.acreate(
                        model=config.default_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        timeout=config.timeout
                    )
                    return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"AI provider {provider} failed", error=str(e))
                continue
        
        raise HTTPException(status_code=500, detail="All AI providers failed")

# === ONYX CACHE MANAGER ===
class OnyxCacheManager:
    """Onyx-optimized cache manager."""
    
    def __init__(self) -> Any:
        self.memory_cache = {}
        self.redis_client = None
        self.stats = {"hits": 0, "misses": 0, "sets": 0}
    
    async def initialize(self) -> Any:
        """Initialize cache connections."""
        if REDIS_AVAILABLE and config.enable_cache:
            try:
                self.redis_client = await aioredis.from_url(
                    config.redis_url,
                    max_connections=20,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Onyx cache initialized with Redis")
            except Exception as e:
                logger.warning("Redis cache failed", error=str(e))
                self.redis_client = None
    
    def _generate_key(self, data: str) -> str:
        """Generate cache key."""
        return f"onyx:copy:{hashlib.md5(data.encode()).hexdigest()[:16]}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        # Memory cache first
        if key in self.memory_cache:
            self.stats["hits"] += 1
            return self.memory_cache[key]
        
        # Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    if JSON_SPEEDUP > 1.0:
                        result = JSON_LIB.loads(cached_data)
                    else:
                        result = JSON_LIB.loads(cached_data)
                    
                    self.memory_cache[key] = result
                    self.stats["hits"] += 1
                    return result
            except Exception as e:
                logger.warning("Cache get failed", error=str(e))
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in cache."""
        try:
            # Memory cache
            self.memory_cache[key] = value
            
            # Redis cache
            if self.redis_client:
                if JSON_SPEEDUP > 1.0:
                    data = JSON_LIB.dumps(value)
                else:
                    data = JSON_LIB.dumps(value)
                
                await self.redis_client.setex(key, ttl or config.cache_ttl, data)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.warning("Cache set failed", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "hit_rate_percent": round(hit_rate, 2),
            "memory_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
            "stats": self.stats
        }

# === ONYX PROMPT MANAGER ===
class OnyxPromptManager:
    """Advanced prompt management with LangChain integration."""
    
    def __init__(self) -> Any:
        self.prompt_templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self) -> Any:
        """Initialize copywriting prompt templates."""
        
        # Base system prompt
        self.system_prompt = """Eres un experto copywriter especializado en marketing digital. 
Tu objetivo es crear contenido persuasivo, atractivo y optimizado para diferentes plataformas.

Características del contenido a generar:
- Persuasivo y orientado a la acción
- Adaptado a la plataforma específica
- Tono apropiado para la audiencia
- Optimizado para engagement
- Culturalmente relevante para el mercado hispanohablante

Siempre mantén un enfoque profesional y ético en tus recomendaciones."""
        
        # Platform-specific templates
        self.prompt_templates = {
            "instagram": {
                "social_post": """
Crea un post para Instagram sobre: {product_description}

Tono: {tone}
Caso de uso: {use_case}
Audiencia: {target_audience}

Formato requerido:
- Headline atractivo (máximo 125 caracteres)
- Texto principal (máximo 2200 caracteres)
- Call-to-action claro
- 5-8 hashtags relevantes
- Emojis apropiados

El contenido debe ser visualmente atractivo y optimizado para engagement.
""",
                "story": """
Crea contenido para Instagram Story sobre: {product_description}

Tono: {tone}
Características:
- Texto breve y impactante
- Llamada a la acción clara
- Optimizado para formato vertical
- Uso estratégico de emojis
"""
            },
            
            "facebook": {
                "social_post": """
Crea un post para Facebook sobre: {product_description}

Tono: {tone}
Caso de uso: {use_case}

Formato requerido:
- Headline atractivo
- Texto principal detallado (máximo 500 palabras)
- Call-to-action persuasivo
- Hashtags estratégicos (3-5)

El contenido debe generar conversación y shares.
""",
                "ad_copy": """
Crea un anuncio para Facebook Ads sobre: {product_description}

Tono: {tone}
Objetivo: {use_case}

Formato requerido:
- Headline principal (máximo 40 caracteres)
- Texto del anuncio (máximo 125 caracteres)
- Descripción (máximo 30 caracteres)
- Call-to-action específico

Optimizado para conversiones y CTR alto.
"""
            },
            
            "twitter": {
                "social_post": """
Crea un tweet sobre: {product_description}

Tono: {tone}
Restricciones:
- Máximo 280 caracteres
- Incluir 2-3 hashtags relevantes
- Call-to-action conciso
- Emojis estratégicos

El tweet debe ser viral y generar engagement.
""",
                "thread": """
Crea un hilo de Twitter (3-5 tweets) sobre: {product_description}

Tono: {tone}
Estructura:
1. Tweet gancho inicial
2-4. Tweets de desarrollo/valor
5. Tweet de cierre con CTA

Cada tweet máximo 280 caracteres.
"""
            },
            
            "linkedin": {
                "social_post": """
Crea un post profesional para LinkedIn sobre: {product_description}

Tono: {tone}
Audiencia: Profesionales y empresarios

Formato:
- Apertura atractiva
- Desarrollo con valor profesional
- Call-to-action profesional
- Hashtags de industria (3-5)

El contenido debe posicionar expertise y generar networking.
""",
                "article": """
Crea un artículo para LinkedIn sobre: {product_description}

Tono: {tone}
Estructura:
- Título atractivo
- Introducción con hook
- 3-4 puntos clave con valor
- Conclusión con CTA
- Longitud: 500-800 palabras

Enfoque en thought leadership y valor profesional.
"""
            },
            
            "email": {
                "subject_line": """
Crea 5 líneas de asunto para email sobre: {product_description}

Tono: {tone}
Caso de uso: {use_case}

Características:
- Máximo 50 caracteres
- Generan curiosidad
- Evitan spam filters
- Orientadas a apertura

Incluye variaciones para A/B testing.
""",
                "email_body": """
Crea el cuerpo de un email sobre: {product_description}

Tono: {tone}
Estructura:
- Saludo personalizado
- Apertura con valor
- Desarrollo del beneficio
- Prueba social/credibilidad
- Call-to-action claro
- Cierre profesional

Longitud: 200-400 palabras, optimizado para conversión.
"""
            }
        }
    
    def get_prompt(self, platform: str, content_type: str, **kwargs) -> str:
        """Get formatted prompt for specific platform and content type."""
        
        # Get template
        template_key = self.prompt_templates.get(platform, {}).get(content_type)
        if not template_key:
            # Fallback to generic template
            template_key = """
Crea contenido de copywriting sobre: {product_description}

Plataforma: {platform}
Tipo de contenido: {content_type}
Tono: {tone}
Caso de uso: {use_case}

Genera contenido persuasivo y optimizado para la plataforma especificada.
"""f"
        
        # Format template with provided data
        try:
            if LANGCHAIN_AVAILABLE:
                prompt_template = PromptTemplate.from_template(template_key)
                return prompt_template.format(**kwargs)
            else:
                return template_key"
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template_key
    
    def get_system_message(self) -> str:
        """Get system message for AI context."""
        return self.system_prompt

# === ONYX COPYWRITING SERVICE ===
class OnyxCopywritingService:
    """Main Onyx-integrated copywriting service."""
    
    def __init__(self) -> Any:
        self.ai_manager = AIProviderManager()
        self.cache_manager = OnyxCacheManager()
        self.prompt_manager = OnyxPromptManager()
        
        self.performance_stats = {
            "requests_processed": 0,
            "total_generation_time": 0.0,
            "ai_calls": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        logger.info("OnyxCopywritingService initialized",
                   performance_level=config.performance_level,
                   ai_providers=len(self.ai_manager.providers))
    
    async def initialize(self) -> Any:
        """Initialize the service."""
        await self.cache_manager.initialize()
        logger.info("Onyx copywriting service initialized")
    
    async def generate_copy(self, input_data: CopywritingInput) -> CopywritingOutput:
        """Generate copywriting content using AI providers."""
        start_time = time.perf_counter()
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Generate cache key
            cache_key = self._generate_cache_key(input_data)
            
            # Check cache
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                logger.info("Cache hit", tracking_id=input_data.tracking_id)
                return CopywritingOutput(**cached_result)
            
            # Generate variants using AI
            variants = await self._generate_ai_variants(input_data)
            
            # Post-process variants
            await self._post_process_variants(variants, input_data)
            
            # Select best variant
            best_variant_id = self._select_best_variant(variants)
            
            # Create output
            generation_time = time.perf_counter() - start_time
            output = CopywritingOutput(
                variants=variants,
                model_used=f"onyx-{self.ai_manager.current_provider}-v1",
                generation_time=generation_time,
                best_variant_id=best_variant_id,
                confidence_score=self._calculate_confidence(variants),
                tracking_id=input_data.tracking_id,
                created_at=datetime.now(timezone.utc),
                performance_metrics={
                    "generation_time_ms": generation_time * 1000,
                    "ai_provider": self.ai_manager.current_provider,
                    "performance_level": config.performance_level,
                    "cache_hit": False,
                    "variants_generated": len(variants)
                }
            )
            
            # Cache result
            asyncio.create_task(
                self.cache_manager.set(cache_key, output.model_dump())
            )
            
            # Update stats
            self._update_stats(generation_time)
            
            return output
            
        except Exception as e:
            self.performance_stats["errors"] += 1
            logger.error("Copy generation failed", error=str(e), tracking_id=input_data.tracking_id)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    def _validate_input(self, input_data: CopywritingInput):
        """Validate input data."""
        if not input_data.product_description or len(input_data.product_description.strip()) == 0:
            raise HTTPException(status_code=400, detail="Product description is required")
        
        if len(input_data.product_description) > 2000:
            raise HTTPException(status_code=400, detail="Product description too long (max 2000 chars)")
    
    def _generate_cache_key(self, input_data: CopywritingInput) -> str:
        """Generate cache key for input."""
        key_parts = [
            input_data.product_description[:100],
            input_data.target_platform.value,
            input_data.content_type.value,
            input_data.tone.value,
            input_data.use_case.value,
            input_data.language.value,
            str(input_data.effective_creativity_score),
            str(input_data.effective_max_variants)
        ]
        
        key_string = "|".join(key_parts)
        return self.cache_manager._generate_key(key_string)
    
    async def _generate_ai_variants(self, input_data: CopywritingInput) -> List[CopyVariant]:
        """Generate variants using AI providers."""
        max_variants = min(input_data.effective_max_variants, 5)
        variants = []
        
        for i in range(max_variants):
            try:
                # Prepare prompt
                prompt_data = {
                    "product_description": input_data.product_description,
                    "platform": input_data.target_platform.value,
                    "content_type": input_data.content_type.value,
                    "tone": input_data.tone.value,
                    "use_case": input_data.use_case.value,
                    "target_audience": input_data.target_audience or "audiencia general",
                    "creativity": input_data.effective_creativity_score
                }
                
                # Get platform-specific prompt
                prompt = self.prompt_manager.get_prompt(
                    input_data.target_platform.value,
                    input_data.content_type.value,
                    **prompt_data
                )
                
                # Add system context
                full_prompt = f"{self.prompt_manager.get_system_message()}\n\n{prompt}"
                
                # Call AI with fallback
                ai_response = await self.ai_manager.call_ai_with_fallback(full_prompt)
                
                # Parse AI response
                variant = self._parse_ai_response(ai_response, input_data, i)
                variants.append(variant)
                
                self.performance_stats["ai_calls"] += 1
                
            except Exception as e:
                logger.warning(f"AI variant generation failed for variant {i}", error=str(e))
                # Create fallback variant
                fallback_variant = self._create_fallback_variant(input_data, i)
                variants.append(fallback_variant)
        
        # Ensure at least one variant
        if not variants:
            variants = [self._create_fallback_variant(input_data, 0)]
        
        return variants
    
    def _parse_ai_response(self, ai_response: str, input_data: CopywritingInput, variant_index: int) -> CopyVariant:
        """Parse AI response into CopyVariant."""
        
        # Simple parsing - extract headline, text, CTA, hashtags
        lines = ai_response.strip().split('\n')
        
        headline = ""
        primary_text = ""
        call_to_action = ""
        hashtags = []
        
        # Basic parsing logic
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#') and not headline:
                headline = line.replace('#', '').strip()
            elif line.startswith('CTA:') or line.startswith('Call to action:'):
                call_to_action = line.split(':', 1)[1].strip()
            elif line.startswith('#') and headline:
                hashtags.append(line.strip())
            elif not headline and len(line) < 200:
                headline = line
            else:
                primary_text += f"{line} "
        
        # Fallback extraction
        if not headline:
            headline = lines[0][:200] if lines else "Contenido generado"
        
        if not primary_text:
            primary_text = ai_response[:1500]
        
        if not call_to_action:
            cta_options = ["¡Descúbrelo ahora!", "Más información", "¡Pruébalo!", "Contacta"]
            call_to_action = cta_options[variant_index % len(cta_options)]
        
        full_text = f"{headline} {primary_text}"
        
        return CopyVariant(
            variant_id=f"{input_data.tracking_id}_onyx_{variant_index}_{int(time.time())}",
            headline=headline[:200],
            primary_text=primary_text.strip()[:1500],
            call_to_action=call_to_action,
            hashtags=hashtags[:8],
            character_count=len(full_text),
            word_count=len(full_text.split()),
            created_at=datetime.now(timezone.utc)
        )
    
    async def _post_process_variants(self, variants: List[CopyVariant], input_data: CopywritingInput):
        """Post-process variants with metrics and optimization."""
        for variant in variants:
            # Calculate basic metrics
            full_text = f"{variant.headline} {variant.primary_text}"
            words = full_text.split()
            
            # Simple readability calculation
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
            readability = max(0, min(100, 100 - (avg_word_length * 8)))
            
            # Engagement prediction based on length and readability
            optimal_length = 50
            length_factor = 1 - abs(len(words) - optimal_length) / optimal_length
            engagement = max(0, min(1, (readability / 100 * 0.6) + (max(0, length_factor) * 0.4)))
            
            variant.readability_score = readability
            variant.engagement_prediction = engagement
    
    def _select_best_variant(self, variants: List[CopyVariant]) -> str:
        """Select the best performing variant."""
        if not variants:
            return ""
        
        def score_variant(variant: CopyVariant) -> float:
            engagement = variant.engagement_prediction or 0
            readability = (variant.readability_score or 0) / 100
            return (engagement * 0.7) + (readability * 0.3)
        
        best_variant = max(variants, key=score_variant)
        return best_variant.variant_id
    
    def _calculate_confidence(self, variants: List[CopyVariant]) -> float:
        """Calculate confidence score."""
        if not variants:
            return 0.0
        
        scores = [v.engagement_prediction or 0 for v in variants]
        avg_score = sum(scores) / len(scores)
        
        # Boost confidence for AI-generated content
        return max(0.0, min(1.0, avg_score * 1.2))
    
    def _create_fallback_variant(self, input_data: CopywritingInput, variant_index: int) -> CopyVariant:
        """Create fallback variant when AI fails."""
        product_name = input_data.product_description.split('.')[0][:50].strip()
        
        return CopyVariant(
            variant_id=f"{input_data.tracking_id}_fallback_{variant_index}",
            headline=f"Descubre {product_name}",
            primary_text=f"La mejor solución para ti. {input_data.product_description[:100]}...",
            call_to_action="Más Información",
            character_count=150,
            word_count=20,
            created_at=datetime.now(timezone.utc)
        )
    
    def _update_stats(self, generation_time: float):
        """Update performance statistics."""
        self.performance_stats["requests_processed"] += 1
        self.performance_stats["total_generation_time"] += generation_time
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        cache_stats = self.cache_manager.get_stats()
        
        avg_time = 0.0
        if self.performance_stats["requests_processed"] > 0:
            avg_time = (
                self.performance_stats["total_generation_time"] / 
                self.performance_stats["requests_processed"]
            )
        
        return {
            "service_stats": self.performance_stats,
            "cache_stats": cache_stats,
            "ai_providers": {
                "available": list(self.ai_manager.providers.keys()),
                "current": self.ai_manager.current_provider,
                "fallbacks": self.ai_manager.fallback_providers
            },
            "performance": {
                "level": config.performance_level,
                "avg_generation_time_ms": avg_time * 1000,
                "json_speedup": f"{JSON_SPEEDUP}x",
                "uvloop_enabled": UVLOOP_AVAILABLE,
                "langchain_available": LANGCHAIN_AVAILABLE
            }
        }

# Global service instance
_onyx_service: Optional[OnyxCopywritingService] = None

async def get_onyx_service() -> OnyxCopywritingService:
    """Get Onyx service instance."""
    global _onyx_service
    if _onyx_service is None:
        _onyx_service = OnyxCopywritingService()
        await _onyx_service.initialize()
    return _onyx_service

# === FASTAPI APPLICATION ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting Onyx-Optimized Copywriting Service",
               performance_level=config.performance_level)
    
    # Set uvloop if available
    if UVLOOP_AVAILABLE and sys.platform != 'win32':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("UVLoop enabled for Onyx service")
    
    # Initialize service
    await get_onyx_service()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Onyx Copywriting Service")

def create_onyx_app() -> FastAPI:
    """Create Onyx-integrated FastAPI application."""
    
    app = FastAPI(
        title="Onyx-Optimized Copywriting Service",
        description=f"""
        **Production Copywriting API for Onyx Backend**
        
        🔧 **Performance Level**: {config.performance_level}
        🤖 **AI Integration**: LangChain + OpenRouter
        ⚡ **Optimizations**: {JSON_SPEEDUP:.1f}x JSON, UVLoop, Redis
        
        ## AI Providers
        - OpenRouter (Multiple models)
        - OpenAI (GPT-3.5, GPT-4)
        - Anthropic (Claude)
        - Automatic fallback system
        
        ## Features
        - Multi-platform content generation
        - LangChain prompt management
        - Intelligent caching system
        - Performance monitoring
        - Onyx backend integration
        - Production-ready optimization
        """,
        version="1.0.0-onyx",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    # === ROUTES ===
    
    @app.get("/")
    async def root():
        """Onyx service information."""
        service = await get_onyx_service()
        stats = await service.get_service_stats()
        
        return {
            "service": "Onyx-Optimized Copywriting Service",
            "version": "1.0.0-onyx",
            "status": "operational",
            "performance": stats["performance"],
            "ai_providers": stats["ai_providers"],
            "features": {
                "langchain_integration": LANGCHAIN_AVAILABLE,
                "openrouter_support": config.openrouter_api_key is not None,
                "multi_provider_fallback": True,
                "intelligent_caching": True,
                "onyx_integration": True
            },
            "endpoints": {
                "generate": "/onyx/generate",
                "health": "/onyx/health",
                "stats": "/onyx/stats",
                "providers": "/onyx/providers"
            }
        }
    
    @app.post("/onyx/generate", response_model=CopywritingOutput)
    async def generate_onyx_copy(
        input_data: CopywritingInput = Body(..., example={
            "product_description": "Plataforma de marketing digital con IA que automatiza campañas publicitarias",
            "target_platform": "instagram",
            "content_type": "social_post",
            "tone": "professional",
            "use_case": "brand_awareness",
            "language": "es",
            "creativity_level": "creative",
            "target_audience": "emprendedores y marketers digitales",
            "website_info": {
                "website_name": "MarketingAI Pro",
                "about": "Automatizamos el marketing digital para empresas",
                "features": ["Automatización", "Analytics", "Personalización"]
            },
            "variant_settings": {
                "max_variants": 3,
                "variant_diversity": 0.8
            }
        })
    ):
        """Generate optimized copywriting content with Onyx integration."""
        service = await get_onyx_service()
        return await service.generate_copy(input_data)
    
    @app.get("/onyx/health")
    async def health_check():
        """Comprehensive health check."""
        service = await get_onyx_service()
        stats = await service.get_service_stats()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "performance": stats["performance"],
            "ai_providers": {
                "available": len(stats["ai_providers"]["available"]),
                "current": stats["ai_providers"]["current"]
            },
            "cache": {
                "hit_rate": stats["cache_stats"]["hit_rate_percent"],
                "redis_connected": stats["cache_stats"]["redis_connected"]
            },
            "requests_processed": stats["service_stats"]["requests_processed"]
        }
    
    @app.get("/onyx/stats")
    async def get_detailed_stats():
        """Get detailed service statistics."""
        service = await get_onyx_service()
        return await service.get_service_stats()
    
    @app.get("/onyx/providers")
    async def get_ai_providers():
        """Get available AI providers information."""
        service = await get_onyx_service()
        
        return {
            "providers": service.ai_manager.providers,
            "current_provider": service.ai_manager.current_provider,
            "fallback_providers": service.ai_manager.fallback_providers,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "openai_available": OPENAI_AVAILABLE
        }
    
    return app

# Create the Onyx application
onyx_app = create_onyx_app()

# === MAIN ===
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Onyx-Optimized Copywriting Service")
    
    uvicorn.run(
        "onyx_optimized:onyx_app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info",
        loop="uvloop" if UVLOOP_AVAILABLE and sys.platform != 'win32' else "asyncio"
    )

# Export
__all__ = [
    "onyx_app", "create_onyx_app", "OnyxCopywritingService",
    "get_onyx_service", "OnyxConfig", "AIProviderManager"
] 