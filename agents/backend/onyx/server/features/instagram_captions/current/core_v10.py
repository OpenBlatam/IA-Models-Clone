"""
Instagram Captions API v10.0 - Refactored Core Module

Consolidates ultra-advanced v9.0 capabilities into a clean, maintainable architecture.
Essential libraries only, maximum performance, simplified deployment.
"""

import asyncio
import time
import json
import hashlib
import secrets
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from typing import Any, List, Dict, Optional, Union, Tuple

# Core framework (essential only)
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import BaseModel, Field, field_validator

# Essential AI libraries (curated from v9.0)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import orjson
    json_dumps = lambda obj: orjson.dumps(obj).decode()
    json_loads = orjson.loads
    ULTRA_JSON = True
except ImportError:
    json_dumps = json.dumps
    json_loads = json.dumps
    ULTRA_JSON = False

try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs) -> Any:
        def decorator(func) -> Any:
            return func
        return decorator

# Performance optimization
try:
    from cachetools import LRUCache, TTLCache
    ADVANCED_CACHE = True
except ImportError:
    ADVANCED_CACHE = False

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_CONNECTIONS = 1000
MAX_RETRIES = 100
TIMEOUT_SECONDS = 60

# =============================================================================
# REFACTORED CONFIGURATION
# =============================================================================

class RefactoredConfig(BaseSettings):
    """Simplified but powerful configuration for v10.0."""
    
    # API Information
    API_VERSION: str = "10.0.0"
    API_NAME: str = "Instagram Captions API v10.0 - Refactored Ultra-Advanced"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8100, env="PORT")  # Dedicated port for v10.0
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    API_KEY_HEADER: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # AI Configuration
    AI_MODEL_NAME: str = Field(default="gpt2", env="AI_MODEL_NAME")
    MAX_TOKENS: int = Field(default=150, env="MAX_TOKENS")
    TEMPERATURE: float = Field(default=0.7, env="TEMPERATURE")
    
    # Performance
    CACHE_SIZE: int = Field(default=1000, env="CACHE_SIZE")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=20, env="RATE_LIMIT_BURST")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# =============================================================================
# REFACTORED SCHEMAS
# =============================================================================

class RefactoredCaptionRequest(BaseModel):
    """Optimized request schema for caption generation."""
    
    text: str = Field(..., min_length=1, max_length=1000, description="Input text for caption generation")
    style: Optional[str] = Field(default="casual", description="Caption style (casual, formal, creative, etc.)")
    length: Optional[str] = Field(default="medium", description="Caption length (short, medium, long)")
    hashtags: Optional[bool] = Field(default=True, description="Include hashtags")
    emojis: Optional[bool] = Field(default=True, description="Include emojis")
    language: Optional[str] = Field(default="en", description="Language code")
    
    @field_validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()
    
    @field_validator('style')
    def validate_style(cls, v):
        valid_styles = ['casual', 'formal', 'creative', 'professional', 'funny', 'inspirational']
        if v not in valid_styles:
            raise ValueError(f'Style must be one of: {", ".join(valid_styles)}')
        return v
    
    @field_validator('length')
    def validate_length(cls, v):
        valid_lengths = ['short', 'medium', 'long']
        if v not in valid_lengths:
            raise ValueError(f'Length must be one of: {", ".join(valid_lengths)}')
        return v

class RefactoredCaptionResponse(BaseModel):
    """Optimized response schema for caption generation."""
    
    caption: str = Field(..., description="Generated Instagram caption")
    style: str = Field(..., description="Applied style")
    length: str = Field(..., description="Applied length")
    hashtags: List[str] = Field(default_factory=list, description="Generated hashtags")
    emojis: List[str] = Field(default_factory=list, description="Used emojis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="AI model used for generation")

class BatchRefactoredRequest(BaseModel):
    """Batch request schema for multiple caption generation."""
    
    requests: List[RefactoredCaptionRequest] = Field(..., min_items=1, max_items=100, description="List of caption requests")
    batch_id: Optional[str] = Field(default_factory=lambda: secrets.token_urlsafe(8), description="Unique batch identifier")
    
    @field_validator('requests')
    def validate_requests(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 requests allowed per batch')
        return v

# =============================================================================
# REFACTORED AI ENGINE
# =============================================================================

class RefactoredAIEngine:
    """Optimized AI engine for caption generation."""
    
    def __init__(self, config: RefactoredConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = TTLCache(maxsize=config.CACHE_SIZE, ttl=config.CACHE_TTL) if ADVANCED_CACHE else {}
        
        # Initialize AI models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models with error handling."""
        try:
            if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.AI_MODEL_NAME)
                self.model = AutoModelForCausalLM.from_pretrained(self.config.AI_MODEL_NAME)
                self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
                self.logger.info("AI models initialized successfully")
            else:
                self.logger.warning("AI libraries not available, using fallback methods")
                self.pipeline = None
        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}")
            self.pipeline = None
    
    async def generate_caption(self, request: RefactoredCaptionRequest) -> RefactoredCaptionResponse:
        """Generate Instagram caption using AI."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.cache:
                cached_response = self.cache[cache_key]
                cached_response.processing_time = time.time() - start_time
                return cached_response
            
            # Generate caption
            caption = await self._generate_text(request)
            hashtags = self._generate_hashtags(request.text, request.style) if request.hashtags else []
            emojis = self._generate_emojis(request.style) if request.emojis else []
            
            response = RefactoredCaptionResponse(
                caption=caption,
                style=request.style,
                length=request.length,
                hashtags=hashtags,
                emojis=emojis,
                metadata={"cache_hit": False},
                processing_time=time.time() - start_time,
                model_used=self.config.AI_MODEL_NAME
            )
            
            # Cache the response
            if ADVANCED_CACHE:
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            raise
    
    async def _generate_text(self, request: RefactoredCaptionRequest) -> str:
        """Generate text using AI model or fallback."""
        if self.pipeline:
            try:
                prompt = self._build_prompt(request)
                result = self.pipeline(
                    prompt,
                    max_length=len(prompt.split()) + self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                return result[0]['generated_text'].replace(prompt, '').strip()
            except Exception as e:
                self.logger.warning(f"AI generation failed, using fallback: {e}")
        
        # Fallback generation
        return self._fallback_generation(request)
    
    def _build_prompt(self, request: RefactoredCaptionRequest) -> str:
        """Build prompt for AI model."""
        style_guide = {
            'casual': 'Write a casual, friendly Instagram caption',
            'formal': 'Write a professional, formal Instagram caption',
            'creative': 'Write a creative, artistic Instagram caption',
            'professional': 'Write a business-oriented Instagram caption',
            'funny': 'Write a humorous, entertaining Instagram caption',
            'inspirational': 'Write an inspiring, motivational Instagram caption'
        }
        
        length_guide = {
            'short': 'Keep it under 50 characters',
            'medium': 'Keep it between 50-150 characters',
            'long': 'Make it detailed, up to 300 characters'
        }
        
        prompt = f"{style_guide.get(request.style, 'Write an Instagram caption')} "
        prompt += f"{length_guide.get(request.length, '')} "
        prompt += f"for this text: '{request.text}'"
        
        return prompt
    
    def _fallback_generation(self, request: RefactoredCaptionRequest) -> str:
        """Fallback caption generation when AI is not available."""
        base_caption = f"📸 {request.text}"
        
        if request.style == 'casual':
            base_caption += " ✨"
        elif request.style == 'formal':
            base_caption += " 📊"
        elif request.style == 'creative':
            base_caption += " 🎨"
        elif request.style == 'professional':
            base_caption += " 💼"
        elif request.style == 'funny':
            base_caption += " 😂"
        elif request.style == 'inspirational':
            base_caption += " 💪"
        
        return base_caption
    
    def _generate_hashtags(self, text: str, style: str) -> List[str]:
        """Generate relevant hashtags."""
        # Simple hashtag generation logic
        words = text.lower().split()
        hashtags = []
        
        # Add style-based hashtags
        style_tags = {
            'casual': ['#casual', '#lifestyle', '#daily'],
            'formal': ['#professional', '#business', '#career'],
            'creative': ['#creative', '#art', '#design'],
            'professional': ['#professional', '#business', '#success'],
            'funny': ['#funny', '#humor', '#laugh'],
            'inspirational': ['#inspiration', '#motivation', '#goals']
        }
        
        hashtags.extend(style_tags.get(style, ['#instagram', '#caption']))
        
        # Add content-based hashtags (first 3 words)
        for word in words[:3]:
            if len(word) > 3 and word.isalpha():
                hashtags.append(f"#{word}")
        
        return hashtags[:10]  # Limit to 10 hashtags
    
    def _generate_emojis(self, style: str) -> List[str]:
        """Generate relevant emojis."""
        emoji_map = {
            'casual': ['✨', '💫', '🌟'],
            'formal': ['📊', '📈', '💼'],
            'creative': ['🎨', '🎭', '🎪'],
            'professional': ['💼', '📋', '🎯'],
            'funny': ['😂', '🤣', '😄'],
            'inspirational': ['💪', '🔥', '🚀']
        }
        
        return emoji_map.get(style, ['📸', '✨'])
    
    def _generate_cache_key(self, request: RefactoredCaptionRequest) -> str:
        """Generate cache key for request."""
        content = f"{request.text}:{request.style}:{request.length}:{request.hashtags}:{request.emojis}:{request.language}"
        return hashlib.md5(content.encode()).hexdigest()

# =============================================================================
# REFACTORED UTILITIES
# =============================================================================

class RefactoredUtils:
    """Utility functions for the refactored system."""
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format and content."""
        if not api_key or len(api_key) < 32:
            return False
        
        # Add more validation logic as needed
        return True
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize input text for safety."""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '(', ')', '{', '}', '[', ']']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text.strip()
    
    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request identifier."""
        timestamp = int(time.time() * 1000)
        random_part = secrets.token_hex(4)
        return f"req_{timestamp}_{random_part}"
    
    @staticmethod
    def format_processing_time(seconds: float) -> str:
        """Format processing time for display."""
        if seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        return f"{seconds:.2f}s"

# =============================================================================
# METRICS AND MONITORING
# =============================================================================

class Metrics:
    """Simple metrics collection for monitoring."""
    
    def __init__(self):
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
    
    def record_request(self, success: bool, processing_time: float):
        """Record request metrics."""
        self.request_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.total_processing_time += processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        uptime = time.time() - self.start_time
        avg_processing_time = self.total_processing_time / max(self.request_count, 1)
        success_rate = (self.success_count / max(self.request_count, 1)) * 100
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate_percent": round(success_rate, 2),
            "average_processing_time": round(avg_processing_time, 3),
            "requests_per_second": round(self.request_count / max(uptime, 1), 2)
        }

# =============================================================================
# AI PROVIDER ENUM
# =============================================================================

class AIProvider(str, Enum):
    """Available AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    FALLBACK = "fallback"

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RefactoredConfig',
    'RefactoredCaptionRequest', 
    'RefactoredCaptionResponse',
    'BatchRefactoredRequest',
    'RefactoredAIEngine',
    'RefactoredUtils',
    'Metrics',
    'AIProvider',
    'TORCH_AVAILABLE',
    'TRANSFORMERS_AVAILABLE',
    'ULTRA_JSON',
    'NUMBA_AVAILABLE',
    'ADVANCED_CACHE'
]
