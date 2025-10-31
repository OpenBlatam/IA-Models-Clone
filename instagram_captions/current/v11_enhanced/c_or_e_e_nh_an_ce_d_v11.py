from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import hashlib
import secrets
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Protocol, TypeVar, Generic
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    from pydantic_settings import BaseSettings
        from pydantic import BaseModel, Field, validator as field_validator, Config
        from pydantic import BaseSettings
    import orjson
    import json
    import numba
    from numba import jit
    from cachetools import TTLCache, LRUCache
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Any, List, Dict, Optional
"""
Instagram Captions API v11.0 - Enhanced Refactor

Advanced refactoring with enterprise-grade patterns, design principles,
and cutting-edge optimizations. The ultimate evolution of the API.
"""


# Enhanced imports with fallbacks
try:
    PYDANTIC_V2 = True
except ImportError:
    try:
        PYDANTIC_V2 = False
    except ImportError:
        PYDANTIC_V2 = False

# Performance libraries
try:
    json_dumps = lambda obj: orjson.dumps(obj).decode()
    json_loads = orjson.loads
    ULTRA_JSON = True
except ImportError:
    json_dumps = json.dumps
    json_loads = json.loads
    ULTRA_JSON = False

try:
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs) -> Any:
        def decorator(func) -> Any:
            return func
        return decorator

try:
    ADVANCED_CACHE = True
except ImportError:
    ADVANCED_CACHE = False

# AI libraries
try:
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DESIGN PATTERNS & ADVANCED ARCHITECTURE
# =============================================================================

T = TypeVar('T')

class Singleton(type):
    """Thread-safe Singleton metaclass for configuration management."""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Observer(Protocol):
    """Observer protocol for event-driven architecture."""
    def update(self, event: str, data: Dict[str, Any]) -> None: ...


class Subject:
    """Subject class for observer pattern implementation."""
    
    def __init__(self) -> Any:
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)
    
    def notify(self, event: str, data: Dict[str, Any]) -> None:
        for observer in self._observers:
            try:
                observer.update(event, data)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")


class AIProvider(ABC):
    """Abstract AI Provider interface using Strategy Pattern."""
    
    @abstractmethod
    async def generate_caption(self, request: 'EnhancedCaptionRequest') -> str:
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        pass


class PerformanceMonitor:
    """Advanced performance monitoring with observer pattern."""
    
    def __init__(self) -> Any:
        self.metrics = {
            'total_requests': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'error_count': 0,
            'last_updated': time.time()
        }
        self._lock = threading.Lock()
    
    def record_request(self, success: bool, response_time: float, cache_hit: bool = False):
        
    """record_request function."""
with self._lock:
            self.metrics['total_requests'] += 1
            
            # Update success rate
            if success:
                success_count = self.metrics['success_rate'] * (self.metrics['total_requests'] - 1) + 1
            else:
                success_count = self.metrics['success_rate'] * (self.metrics['total_requests'] - 1)
                self.metrics['error_count'] += 1
            
            self.metrics['success_rate'] = success_count / self.metrics['total_requests']
            
            # Update response time
            total_time = self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1) + response_time
            self.metrics['avg_response_time'] = total_time / self.metrics['total_requests']
            
            # Update cache hit rate (simplified)
            if cache_hit:
                self.metrics['cache_hit_rate'] = min(self.metrics['cache_hit_rate'] + 0.01, 1.0)
            
            self.metrics['last_updated'] = time.time()
    
    def get_performance_grade(self) -> str:
        """Calculate performance grade based on multiple metrics."""
        success_rate = self.metrics['success_rate']
        response_time = self.metrics['avg_response_time']
        
        if success_rate >= 0.99 and response_time <= 0.05:
            return "A+ ENTERPRISE"
        elif success_rate >= 0.95 and response_time <= 0.1:
            return "A EXCELLENT"
        elif success_rate >= 0.90 and response_time <= 0.2:
            return "B GOOD"
        else:
            return "C NEEDS_IMPROVEMENT"


# =============================================================================
# ENHANCED CONFIGURATION WITH ENTERPRISE FEATURES
# =============================================================================

class EnhancedConfig(BaseSettings, metaclass=Singleton):
    """Enhanced configuration with enterprise features and validation."""
    
    # API Information
    API_VERSION: str = "11.0.0"
    API_NAME: str = "Instagram Captions API v11.0 - Enhanced Enterprise"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8110, env="PORT")  # Dedicated port for v11.0
    
    # Enterprise Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    VALID_API_KEYS: List[str] = Field(default=[
        "enhanced-v11-key", "enterprise-key", "advanced-key"
    ])
    JWT_SECRET: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    
    # Performance Configuration (Enterprise Grade)
    MAX_BATCH_SIZE: int = Field(default=100, env="MAX_BATCH_SIZE")  # Increased for enterprise
    AI_WORKERS: int = Field(default=12, env="AI_WORKERS")  # Optimized worker count
    CACHE_SIZE: int = Field(default=50000, env="CACHE_SIZE")  # Enterprise cache
    CACHE_TTL: int = Field(default=7200, env="CACHE_TTL")  # 2 hours
    
    # AI Configuration (Advanced)
    AI_MODEL: str = Field(default="distilgpt2", env="AI_MODEL")
    AI_TEMPERATURE: float = Field(default=0.8, env="AI_TEMPERATURE")
    AI_MAX_LENGTH: int = Field(default=200, env="AI_MAX_LENGTH")
    ENABLE_GPU: bool = Field(default=False, env="ENABLE_GPU")
    
    # Enterprise Features
    ENABLE_AUDIT_LOG: bool = Field(default=True, env="ENABLE_AUDIT_LOG")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    ENABLE_RATE_LIMITING: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    ENABLE_MULTI_TENANT: bool = Field(default=False, env="ENABLE_MULTI_TENANT")
    
    # Rate Limiting (Advanced)
    RATE_LIMIT_REQUESTS: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    RATE_LIMIT_BURST: int = Field(default=50, env="RATE_LIMIT_BURST")
    
    # Monitoring & Observability
    METRICS_RETENTION_DAYS: int = Field(default=30, env="METRICS_RETENTION_DAYS")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    if PYDANTIC_V2:
        model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"


# Global enhanced configuration
config = EnhancedConfig()


# =============================================================================
# ENHANCED DATA MODELS WITH VALIDATION
# =============================================================================

class CaptionStyle(str, Enum):
    """Enhanced caption styles with more options."""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    INSPIRATIONAL = "inspirational"
    LUXURY = "luxury"
    EDUCATIONAL = "educational"
    STORYTELLING = "storytelling"
    CALL_TO_ACTION = "call_to_action"


class AIProviderType(str, Enum):
    """Available AI providers."""
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    CLAUDE = "claude"
    FALLBACK = "fallback"


class EnhancedCaptionRequest(BaseModel):
    """Enhanced request model with enterprise validation."""
    
    content_description: str = Field(
        ..., 
        min_length=5, 
        max_length=2000,
        description="Detailed content description"
    )
    
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Caption style preference"
    )
    
    hashtag_count: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of hashtags to generate"
    )
    
    ai_provider: AIProviderType = Field(
        default=AIProviderType.TRANSFORMERS,
        description="AI provider to use"
    )
    
    # Enterprise features
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier for multi-tenant support")
    user_id: Optional[str] = Field(default=None, description="User identifier for audit logging")
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    custom_instructions: Optional[str] = Field(default=None, max_length=500)
    
    # Advanced analysis options
    enable_advanced_analysis: bool = Field(default=True)
    include_sentiment_analysis: bool = Field(default=True)
    include_engagement_prediction: bool = Field(default=True)
    include_competitor_analysis: bool = Field(default=False)
    
    # Client information
    client_id: str = Field(default="enhanced-v11")
    client_version: Optional[str] = Field(default=None)
    
    @field_validator('content_description')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Enhanced content validation with security checks."""
        # Security validation
        harmful_patterns = [
            '<script', 'javascript:', 'data:', 'vbscript:', 'onload=', 'onerror='
        ]
        
        for pattern in harmful_patterns:
            if pattern.lower() in v.lower():
                raise ValueError(f"Potentially harmful content detected: {pattern}")
        
        # Content quality validation
        if len(v.strip()) < 5:
            raise ValueError("Content description too short")
        
        return v.strip()


class EnhancedCaptionResponse(BaseModel):
    """Enhanced response model with comprehensive metadata."""
    
    # Core response
    request_id: str
    caption: str
    hashtags: List[str]
    
    # Quality metrics
    quality_score: float = Field(..., ge=0, le=100)
    engagement_prediction: float = Field(..., ge=0, le=100)
    virality_score: float = Field(..., ge=0, le=100)
    
    # Performance metrics
    processing_time: float
    cache_hit: bool = False
    
    # AI metadata
    ai_provider: str
    model_used: str
    confidence_score: float = Field(..., ge=0, le=1)
    
    # Enterprise features
    tenant_id: Optional[str] = None
    audit_id: Optional[str] = None
    
    # Advanced analysis
    advanced_analysis: Optional[Dict[str, Any]] = None
    sentiment_analysis: Optional[Dict[str, Any]] = None
    competitor_insights: Optional[Dict[str, Any]] = None
    
    # Version info
    api_version: str = "11.0.0"
    timestamp: str
    
    # Recommendations
    optimization_suggestions: Optional[List[str]] = None
    alternative_styles: Optional[List[str]] = None


# =============================================================================
# AI PROVIDER FACTORY PATTERN
# =============================================================================

class TransformersAIProvider(AIProvider):
    """Transformers-based AI provider with optimizations."""
    
    def __init__(self) -> Any:
        self.models = {}
        self._initialized = False
        self.stats = {
            "requests_processed": 0,
            "avg_generation_time": 0.0,
            "success_rate": 1.0
        }
    
    async def _initialize_models(self) -> Any:
        """Lazy model initialization."""
        if self._initialized or not AI_AVAILABLE:
            return
        
        try:
            logger.info(f"🤖 Loading {config.AI_MODEL} for enhanced processing...")
            
            self.models['tokenizer'] = AutoTokenizer.from_pretrained(
                config.AI_MODEL,
                pad_token="<|endoftext|>",
                eos_token="<|endoftext|>"
            )
            
            device = "cuda" if config.ENABLE_GPU and torch.cuda.is_available() else "cpu"
            self.models['generator'] = AutoModelForCausalLM.from_pretrained(
                config.AI_MODEL,
                torch_dtype=torch.float32,
                device_map=device
            )
            
            self._initialized = True
            logger.info("✅ Enhanced AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load AI models: {e}")
            raise
    
    async def generate_caption(self, request: EnhancedCaptionRequest) -> str:
        """Generate caption using transformers."""
        await self._initialize_models()
        
        if not self._initialized:
            raise Exception("AI models not available")
        
        # Enhanced style-specific prompts
        style_prompts = {
            CaptionStyle.CASUAL: f"Write a casual, relatable Instagram caption about {request.content_description}:",
            CaptionStyle.PROFESSIONAL: f"Create a professional Instagram caption about {request.content_description}:",
            CaptionStyle.LUXURY: f"Write a luxurious, high-end Instagram caption about {request.content_description}:",
            CaptionStyle.EDUCATIONAL: f"Create an informative Instagram caption about {request.content_description}:",
            CaptionStyle.STORYTELLING: f"Tell a compelling story about {request.content_description}:",
            CaptionStyle.CALL_TO_ACTION: f"Write an engaging call-to-action caption about {request.content_description}:"
        }
        
        prompt = style_prompts.get(request.style, style_prompts[CaptionStyle.CASUAL])
        
        # Add custom instructions if provided
        if request.custom_instructions:
            prompt += f"\n\nAdditional instructions: {request.custom_instructions}"
        
        # Generate with enhanced parameters
        inputs = self.models['tokenizer'].encode(
            prompt,
            return_tensors="pt",
            max_length=100,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.models['generator'].generate(
                inputs,
                max_length=config.AI_MAX_LENGTH,
                temperature=config.AI_TEMPERATURE,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.models['tokenizer'].eos_token_id,
                eos_token_id=self.models['tokenizer'].eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1
            )
        
        # Decode and enhance
        generated_text = self.models['tokenizer'].decode(outputs[0], skip_special_tokens=True)
        caption = generated_text.replace(prompt, "").strip()
        
        # Quality enhancement
        if not caption or len(caption) < 10:
            caption = self._generate_fallback_caption(request)
        
        # Add emoji if missing for certain styles
        if request.style in [CaptionStyle.CASUAL, CaptionStyle.PLAYFUL] and not any(ord(char) > 127 for char in caption):
            caption += " ✨"
        
        # Update stats
        self.stats["requests_processed"] += 1
        
        return caption
    
    def _generate_fallback_caption(self, request: EnhancedCaptionRequest) -> str:
        """Enhanced fallback caption generation."""
        style_templates = {
            CaptionStyle.CASUAL: f"Just discovered this amazing {request.content_description} and had to share! ✨",
            CaptionStyle.PROFESSIONAL: f"Showcasing excellence in {request.content_description} - quality that speaks for itself.",
            CaptionStyle.LUXURY: f"Indulge in the finest {request.content_description} - where luxury meets perfection 💎",
            CaptionStyle.EDUCATIONAL: f"Did you know? This {request.content_description} demonstrates important principles...",
            CaptionStyle.STORYTELLING: f"There's a story behind every {request.content_description}. Let me tell you about this one...",
            CaptionStyle.CALL_TO_ACTION: f"Don't miss out on this incredible {request.content_description}! Tap the link to learn more 👆"
        }
        
        return style_templates.get(request.style, style_templates[CaptionStyle.CASUAL])
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and statistics."""
        return {
            "provider": "transformers",
            "model": config.AI_MODEL,
            "initialized": self._initialized,
            "stats": self.stats,
            "capabilities": ["text_generation", "style_adaptation", "context_awareness"]
        }


class FallbackAIProvider(AIProvider):
    """Fallback AI provider with enhanced templates."""
    
    async def generate_caption(self, request: EnhancedCaptionRequest) -> str:
        """Generate caption using enhanced templates."""
        templates = {
            CaptionStyle.CASUAL: [
                f"Absolutely loving this {request.content_description} moment! ✨",
                f"Can't get enough of this amazing {request.content_description} 🌟",
                f"This {request.content_description} just made my entire day! 💫"
            ],
            CaptionStyle.PROFESSIONAL: [
                f"Excellence in {request.content_description} - setting new standards.",
                f"Professional quality {request.content_description} that delivers results.",
                f"Industry-leading {request.content_description} that speaks for itself."
            ],
            CaptionStyle.LUXURY: [
                f"Indulge in the finest {request.content_description} experience 💎",
                f"Luxury redefined through exceptional {request.content_description} ✨",
                f"Where elegance meets {request.content_description} - pure sophistication 🌟"
            ]
        }
        
        style_templates = templates.get(request.style, templates[CaptionStyle.CASUAL])
        selected = style_templates[hash(request.content_description) % len(style_templates)]
        
        return selected
    
    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "fallback",
            "model": "template_based",
            "capabilities": ["template_generation", "style_variation"]
        }


class AIProviderFactory:
    """Factory for creating AI providers."""
    
    @staticmethod
    def create_provider(provider_type: AIProviderType) -> AIProvider:
        """Create AI provider based on type."""
        if provider_type == AIProviderType.TRANSFORMERS and AI_AVAILABLE:
            return TransformersAIProvider()
        else:
            return FallbackAIProvider()


# =============================================================================
# ENHANCED AI ENGINE WITH ENTERPRISE FEATURES
# =============================================================================

class EnhancedAIEngine(Subject):
    """Enhanced AI engine with enterprise patterns and optimizations."""
    
    def __init__(self) -> Any:
        super().__init__()
        self.provider_factory = AIProviderFactory()
        self.performance_monitor = PerformanceMonitor()
        
        # Enterprise caching
        if ADVANCED_CACHE:
            self.cache = TTLCache(maxsize=config.CACHE_SIZE, ttl=config.CACHE_TTL)
        else:
            self.cache = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_quality": 0.0,
            "avg_processing_time": 0.0,
            "provider_usage": {}
        }
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.AI_WORKERS)
        
        logger.info("🚀 Enhanced AI Engine v11.0 initialized")
    
    def _create_cache_key(self, request: EnhancedCaptionRequest) -> str:
        """Create intelligent cache key."""
        key_data = {
            "content": request.content_description,
            "style": request.style.value,
            "provider": request.ai_provider.value,
            "hashtag_count": request.hashtag_count,
            "custom": request.custom_instructions
        }
        key_string = json_dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def _calculate_enhanced_quality_score(self, caption: str, content: str) -> float:
        """Ultra-fast quality calculation with JIT optimization."""
        base_score = 75.0
        
        # Length optimization (enhanced)
        caption_len = len(caption)
        if 40 <= caption_len <= 200:
            base_score += 15
        elif 30 <= caption_len <= 250:
            base_score += 10
        elif caption_len < 20:
            base_score -= 20
        
        # Word count optimization
        word_count = len(caption.split())
        if 8 <= word_count <= 30:
            base_score += 10
        
        # Engagement indicators (enhanced)
        engagement_words = ['amazing', 'incredible', 'beautiful', 'stunning', 'perfect', 'awesome', 'fantastic']
        engagement_count = sum(1 for word in engagement_words if word in caption.lower())
        base_score += min(engagement_count * 3, 15)
        
        return min(base_score, 100.0)
    
    def _calculate_virality_score(self, caption: str) -> float:
        """Calculate potential virality score."""
        base_virality = 50.0
        
        # Question boost
        if '?' in caption:
            base_virality += 20
        
        # Emoji presence
        emoji_count = sum(1 for char in caption if ord(char) > 127)
        base_virality += min(emoji_count * 5, 25)
        
        # Call-to-action indicators
        cta_words = ['click', 'link', 'bio', 'follow', 'subscribe', 'share', 'comment']
        if any(word in caption.lower() for word in cta_words):
            base_virality += 15
        
        # Trending words (simplified)
        trending_words = ['viral', 'trending', 'must-see', 'exclusive', 'limited']
        if any(word in caption.lower() for word in trending_words):
            base_virality += 10
        
        return min(base_virality, 100.0)
    
    def _generate_enhanced_hashtags(self, request: EnhancedCaptionRequest) -> List[str]:
        """Generate intelligent hashtags with advanced strategy."""
        
        # Enhanced hashtag strategy based on style
        hashtag_strategies = {
            CaptionStyle.CASUAL: {
                "base": ["#lifestyle", "#daily", "#vibes", "#mood", "#authentic", "#real"],
                "engagement": ["#instagood", "#photooftheday", "#love", "#beautiful", "#amazing"],
                "growth": ["#follow", "#like", "#share", "#comment", "#engage"]
            },
            CaptionStyle.PROFESSIONAL: {
                "base": ["#business", "#professional", "#quality", "#excellence", "#success"],
                "engagement": ["#leadership", "#innovation", "#results", "#achievement", "#growth"],
                "industry": ["#expert", "#consulting", "#strategy", "#solutions", "#premium"]
            },
            CaptionStyle.LUXURY: {
                "base": ["#luxury", "#premium", "#exclusive", "#highend", "#sophisticated"],
                "engagement": ["#elegant", "#refined", "#prestige", "#distinction", "#excellence"],
                "lifestyle": ["#luxurylife", "#finestthings", "#exclusive", "#bespoke", "#craftsmanship"]
            },
            CaptionStyle.EDUCATIONAL: {
                "base": ["#education", "#learning", "#knowledge", "#tips", "#howto"],
                "engagement": ["#didyouknow", "#facts", "#insights", "#wisdom", "#growth"],
                "value": ["#valuable", "#useful", "#informative", "#educational", "#helpful"]
            }
        }
        
        strategy = hashtag_strategies.get(request.style, hashtag_strategies[CaptionStyle.CASUAL])
        selected_hashtags = []
        
        # Add base hashtags
        selected_hashtags.extend(strategy["base"][:6])
        
        # Add engagement hashtags
        selected_hashtags.extend(strategy["engagement"][:6])
        
        # Add style-specific hashtags
        if "growth" in strategy:
            selected_hashtags.extend(strategy["growth"][:3])
        elif "industry" in strategy:
            selected_hashtags.extend(strategy["industry"][:3])
        elif "lifestyle" in strategy:
            selected_hashtags.extend(strategy["lifestyle"][:3])
        elif "value" in strategy:
            selected_hashtags.extend(strategy["value"][:3])
        
        # Add content-specific hashtags
        content_words = request.content_description.lower().split()
        for word in content_words:
            if len(word) > 3 and word.isalpha() and len(selected_hashtags) < request.hashtag_count:
                selected_hashtags.append(f"#{word}")
        
        # Fill remaining with trending
        trending_hashtags = ["#viral", "#trending", "#explore", "#discover", "#featured"]
        while len(selected_hashtags) < request.hashtag_count:
            for tag in trending_hashtags:
                if tag not in selected_hashtags and len(selected_hashtags) < request.hashtag_count:
                    selected_hashtags.append(tag)
        
        return selected_hashtags[:request.hashtag_count]
    
    def _perform_advanced_analysis(self, caption: str, request: EnhancedCaptionRequest) -> Dict[str, Any]:
        """Perform comprehensive content analysis."""
        analysis = {
            "word_count": len(caption.split()),
            "character_count": len(caption),
            "sentence_count": len([s for s in caption.split('.') if s.strip()]),
            "readability_score": min(len(caption.split()) / 20, 1.0),  # Simplified
            "has_emoji": any(ord(char) > 127 for char in caption),
            "has_questions": "?" in caption,
            "has_hashtags": "#" in caption,
            "has_mentions": "@" in caption,
            "urgency_indicators": any(word in caption.lower() for word in ['now', 'today', 'urgent', 'limited', 'exclusive']),
            "call_to_action": any(word in caption.lower() for word in ['click', 'link', 'follow', 'subscribe', 'buy']),
            "style_consistency": request.style.value,
            "engagement_potential": "high" if "?" in caption or any(ord(char) > 127 for char in caption) else "medium"
        }
        
        # Sentiment analysis (simplified)
        positive_words = ['amazing', 'beautiful', 'love', 'perfect', 'awesome', 'incredible', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible']
        
        positive_count = sum(1 for word in positive_words if word in caption.lower())
        negative_count = sum(1 for word in negative_words if word in caption.lower())
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        analysis["sentiment"] = {
            "overall": sentiment,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "confidence": min((abs(positive_count - negative_count) + 1) / 5, 1.0)
        }
        
        return analysis
    
    async def generate_enhanced_caption(self, request: EnhancedCaptionRequest) -> EnhancedCaptionResponse:
        """Generate caption with full enterprise features."""
        
        start_time = time.time()
        request_id = f"v11-{int(time.time() * 1000) % 1000000:06d}"
        
        # Check cache
        cache_key = self._create_cache_key(request)
        cache_hit = False
        
        if ADVANCED_CACHE and cache_key in self.cache:
            cached_response = self.cache[cache_key]
            cached_response['request_id'] = request_id
            cached_response['timestamp'] = datetime.now(timezone.utc).isoformat()
            cached_response['cache_hit'] = True
            cache_hit = True
            
            self.stats["cache_hits"] += 1
            logger.info(f"📦 Cache hit for request {request_id}")
            
            return EnhancedCaptionResponse(**cached_response)
        
        try:
            # Create AI provider
            ai_provider = self.provider_factory.create_provider(request.ai_provider)
            
            # Generate caption
            caption = await ai_provider.generate_caption(request)
            
            # Generate hashtags
            hashtags = self._generate_enhanced_hashtags(request)
            
            # Calculate enhanced metrics
            quality_score = self._calculate_enhanced_quality_score(caption, request.content_description)
            virality_score = self._calculate_virality_score(caption)
            engagement_prediction = (quality_score + virality_score) / 2
            
            # Advanced analysis
            advanced_analysis = None
            if request.enable_advanced_analysis:
                advanced_analysis = self._perform_advanced_analysis(caption, request)
            
            # Sentiment analysis
            sentiment_analysis = None
            if request.include_sentiment_analysis and advanced_analysis:
                sentiment_analysis = advanced_analysis["sentiment"]
            
            # Processing metrics
            processing_time = time.time() - start_time
            confidence_score = min(quality_score / 100, 1.0)
            
            # Provider info
            provider_info = ai_provider.get_provider_info()
            
            # Create response
            response_data = {
                "request_id": request_id,
                "caption": caption,
                "hashtags": hashtags,
                "quality_score": quality_score,
                "engagement_prediction": engagement_prediction,
                "virality_score": virality_score,
                "processing_time": processing_time,
                "cache_hit": cache_hit,
                "ai_provider": request.ai_provider.value,
                "model_used": provider_info.get("model", "unknown"),
                "confidence_score": confidence_score,
                "tenant_id": request.tenant_id,
                "advanced_analysis": advanced_analysis,
                "sentiment_analysis": sentiment_analysis,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache response
            if ADVANCED_CACHE:
                self.cache[cache_key] = response_data.copy()
            
            # Update statistics
            self._update_stats(quality_score, processing_time, request.ai_provider.value)
            
            # Record performance
            self.performance_monitor.record_request(True, processing_time, cache_hit)
            
            # Notify observers
            self.notify("caption_generated", {
                "request_id": request_id,
                "quality_score": quality_score,
                "processing_time": processing_time
            })
            
            logger.info(f"✅ Enhanced caption generated {request_id} in {processing_time:.3f}s")
            
            return EnhancedCaptionResponse(**response_data)
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_monitor.record_request(False, processing_time)
            
            logger.error(f"❌ Caption generation failed: {e}")
            
            # Return enhanced fallback
            fallback_provider = FallbackAIProvider()
            fallback_caption = await fallback_provider.generate_caption(request)
            
            return EnhancedCaptionResponse(
                request_id=request_id,
                caption=fallback_caption,
                hashtags=self._generate_enhanced_hashtags(request),
                quality_score=75.0,
                engagement_prediction=60.0,
                virality_score=50.0,
                processing_time=processing_time,
                cache_hit=False,
                ai_provider="fallback",
                model_used="enhanced_fallback",
                confidence_score=0.7,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    def _update_stats(self, quality_score: float, processing_time: float, provider: str):
        """Update engine statistics."""
        self.stats["total_requests"] += 1
        total = self.stats["total_requests"]
        
        # Update averages
        self.stats["avg_quality"] = (
            (self.stats["avg_quality"] * (total - 1) + quality_score) / total
        )
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (total - 1) + processing_time) / total
        )
        
        # Update provider usage
        if provider not in self.stats["provider_usage"]:
            self.stats["provider_usage"][provider] = 0
        self.stats["provider_usage"][provider] += 1
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            "engine_info": {
                "version": "11.0.0",
                "type": "enhanced_enterprise",
                "cache_size": len(self.cache) if hasattr(self.cache, '__len__') else 0,
                "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_requests"], 1)
            },
            "performance": {
                "total_requests": self.stats["total_requests"],
                "avg_quality": self.stats["avg_quality"],
                "avg_processing_time": self.stats["avg_processing_time"],
                "performance_grade": self.performance_monitor.get_performance_grade()
            },
            "providers": self.stats["provider_usage"],
            "capabilities": [
                "advanced_ai_generation",
                "intelligent_caching",
                "enterprise_monitoring",
                "multi_provider_support",
                "enhanced_analytics",
                "style_adaptation",
                "quality_optimization"
            ]
        }


# =============================================================================
# ENHANCED UTILITIES & HELPERS
# =============================================================================

class EnhancedUtils:
    """Enhanced utility functions with enterprise features."""
    
    @staticmethod
    async def generate_request_id(prefix: str = "v11") -> str:
        """Generate enhanced request ID with prefix."""
        timestamp = int(time.time() * 1000000)
        return f"{prefix}-{timestamp % 10000000:07d}"
    
    @staticmethod
    async def validate_api_key(api_key: str) -> bool:
        """Enhanced API key validation."""
        return api_key in config.VALID_API_KEYS
    
    @staticmethod
    def sanitize_content(content: str) -> str:
        """Enhanced content sanitization."""
        harmful_patterns = [
            '<script', 'javascript:', 'data:', 'vbscript:', 
            'onload=', 'onerror=', 'eval(', 'document.cookie'
        ]
        
        for pattern in harmful_patterns:
            content = content.replace(pattern, '')
        
        return content.strip()
    
    @staticmethod
    def format_response_time(seconds: float) -> str:
        """Format response time with enhanced precision."""
        if seconds < 0.001:
            return f"{seconds * 1000000:.0f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        else:
            return f"{seconds:.3f}s"
    
    @staticmethod
    def calculate_rate_limit_remaining(requests_made: int, window_start: float) -> Dict[str, Any]:
        """Calculate rate limiting information."""
        current_time = time.time()
        window_elapsed = current_time - window_start
        
        if window_elapsed >= config.RATE_LIMIT_WINDOW:
            # Reset window
            return {
                "requests_remaining": config.RATE_LIMIT_REQUESTS,
                "reset_time": current_time + config.RATE_LIMIT_WINDOW,
                "window_start": current_time
            }
        else:
            requests_remaining = max(0, config.RATE_LIMIT_REQUESTS - requests_made)
            reset_time = window_start + config.RATE_LIMIT_WINDOW
            
            return {
                "requests_remaining": requests_remaining,
                "reset_time": reset_time,
                "window_start": window_start
            }


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Initialize enhanced global instances
enhanced_ai_engine = EnhancedAIEngine()
enhanced_utils = EnhancedUtils()

# Export enhanced components
__all__ = [
    'config', 'EnhancedCaptionRequest', 'EnhancedCaptionResponse',
    'CaptionStyle', 'AIProviderType', 'enhanced_ai_engine', 
    'EnhancedUtils', 'PerformanceMonitor', 'AIProviderFactory'
] 