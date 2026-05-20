from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import uuid
import logging
import re
    import openai
    from langchain.llms import OpenAI as LangChainOpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import HumanMessage, SystemMessage
    import requests
from .models import (
from .core import InstagramCaptionsEngine
from .gmt_system import SimplifiedGMTSystem
import requests
        from collections import Counter
from typing import Any, List, Dict, Optional
"""
Instagram Captions Service.

Main service for generating Instagram captions with AI provider integrations.
"""


# AI Provider imports
try:
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Local imports
    InstagramCaptionRequest,
    InstagramCaptionResponse,
    CaptionVariation,
    CaptionStyle,
    InstagramTarget,
    HashtagStrategy,
    ContentType,
    TimeZone,
    GenerationMetrics,
    RegionalAdaptation,
    TimeZoneInfo
)

logger = logging.getLogger(__name__)

class AIProvider:
    """Base AI provider interface."""
    
    def __init__(self, name: str):
        
    """__init__ function."""
self.name = name
        self.available = False
        
    async def generate_caption(self, prompt: str, **kwargs) -> str:
        """Generate caption using this provider."""
        raise NotImplementedError
        
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self.available

class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""
    
    def __init__(self) -> Any:
        super().__init__("openai")
        self.client = None
        self.available = OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))
        
        if self.available:
            try:
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.client = openai
                logger.info("OpenAI provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.available = False
    
    async def generate_caption(self, prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> str:
        """Generate caption using OpenAI."""
        if not self.available:
            raise ValueError("OpenAI provider not available")
            
        try:
            response = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self.client.ChatCompletion.create,
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert Instagram caption writer that creates engaging, authentic content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

class LangChainProvider(AIProvider):
    """LangChain provider implementation."""
    
    def __init__(self) -> Any:
        super().__init__("langchain")
        self.llm = None
        self.chain = None
        self.available = LANGCHAIN_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))
        
        if self.available:
            try:
                self.llm = ChatOpenAI(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="gpt-3.5-turbo",
                    temperature=0.7
                )
                
                # Create prompt template
                prompt_template = ChatPromptTemplate.from_messages([
                    SystemMessage(content="You are an expert Instagram caption writer. Create engaging, authentic captions that drive engagement."),
                    HumanMessage(content="{prompt}")
                ])
                
                self.chain = LLMChain(
                    llm=self.llm,
                    prompt=prompt_template
                )
                
                logger.info("LangChain provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain: {e}")
                self.available = False
    
    async def generate_caption(self, prompt: str, **kwargs) -> str:
        """Generate caption using LangChain."""
        if not self.available:
            raise ValueError("LangChain provider not available")
            
        try:
            result = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self.chain.run,
                prompt=prompt
            )
            return result.strip()
            
        except Exception as e:
            logger.error(f"LangChain generation failed: {e}")
            raise

class OpenRouterProvider(AIProvider):
    """OpenRouter provider implementation."""
    
    def __init__(self) -> Any:
        super().__init__("openrouter")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.available = REQUESTS_AVAILABLE and bool(self.api_key)
        
        if self.available:
            logger.info("OpenRouter provider initialized successfully")
    
    async def generate_caption(self, prompt: str, model: str = "anthropic/claude-3-haiku", **kwargs) -> str:
        """Generate caption using OpenRouter."""
        if not self.available:
            raise ValueError("OpenRouter provider not available")
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Instagram Captions Generator"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert Instagram caption writer that creates engaging, authentic content."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": kwargs.get('max_tokens', 500),
                "temperature": kwargs.get('temperature', 0.7)
            }
            
            def make_request():
                
    """make_request function."""
                response = requests.post(f"{self.base_url}/chat/completions", 
                                       headers=headers, 
                                       json=payload, 
                                       timeout=30)
                response.raise_for_status()
                return response.json()
            
            result = await asyncio.to_thread(make_request)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise

class PromptBuilder:
    """Build optimized prompts for Instagram caption generation."""
    
    def __init__(self) -> Any:
        self.style_prompts = {
            CaptionStyle.CASUAL: "Write in a relaxed, conversational tone like talking to a friend",
            CaptionStyle.PROFESSIONAL: "Use professional, polished language suitable for business",
            CaptionStyle.PLAYFUL: "Be fun, energetic, and playful with language",
            CaptionStyle.INSPIRATIONAL: "Create motivational, uplifting content that inspires action",
            CaptionStyle.EDUCATIONAL: "Provide valuable information in an engaging way",
            CaptionStyle.STORYTELLING: "Tell a compelling story that connects emotionally",
            CaptionStyle.PROMOTIONAL: "Create persuasive content that drives action",
            CaptionStyle.MINIMALIST: "Keep it simple, clean, and to the point",
            CaptionStyle.TRENDY: "Use current trends, slang, and pop culture references",
            CaptionStyle.AUTHENTIC: "Be genuine, relatable, and true to voice"
        }
        
        self.audience_prompts = {
            InstagramTarget.MILLENNIALS: "Target millennials (ages 25-40) with nostalgic and career-focused content",
            InstagramTarget.GEN_Z: "Target Gen Z (ages 16-24) with trendy, authentic, and socially conscious content",
            InstagramTarget.BUSINESS: "Target business professionals with valuable, professional content",
            InstagramTarget.CREATORS: "Target content creators with behind-the-scenes and creative insights",
            InstagramTarget.LIFESTYLE: "Target lifestyle enthusiasts with aspirational and relatable content",
            InstagramTarget.FASHION: "Target fashion lovers with style tips and trend insights",
            InstagramTarget.FOOD: "Target food enthusiasts with appetizing descriptions and recipes",
            InstagramTarget.TRAVEL: "Target travelers with wanderlust-inspiring content",
            InstagramTarget.FITNESS: "Target fitness enthusiasts with motivational and instructional content",
            InstagramTarget.TECH: "Target tech enthusiasts with innovative and educational content"
        }
        
        self.hashtag_strategies = {
            HashtagStrategy.TRENDING: "Include current trending hashtags",
            HashtagStrategy.NICHE: "Focus on specific niche hashtags for targeted reach",
            HashtagStrategy.BRANDED: "Emphasize brand-specific and campaign hashtags",
            HashtagStrategy.LOCATION: "Include location-based hashtags",
            HashtagStrategy.MIXED: "Mix trending, niche, and branded hashtags",
            HashtagStrategy.MINIMAL: "Use fewer, highly targeted hashtags",
            HashtagStrategy.AGGRESSIVE: "Maximize hashtag usage for reach"
        }
    
    def build_caption_prompt(self, request: InstagramCaptionRequest) -> str:
        """Build comprehensive prompt for caption generation."""
        prompt_parts = []
        
        # Base instruction
        prompt_parts.append("Generate an engaging Instagram caption based on the following requirements:")
        
        # Content information
        prompt_parts.append(f"\nContent Description: {request.content.description}")
        
        if request.content.image_description:
            prompt_parts.append(f"Image/Visual: {request.content.image_description}")
            
        if request.content.product_info:
            prompt_parts.append(f"Product Info: {request.content.product_info}")
            
        if request.content.occasion:
            prompt_parts.append(f"Occasion: {request.content.occasion}")
            
        if request.content.location:
            prompt_parts.append(f"Location: {request.content.location}")
        
        # Brand information
        if request.brand:
            prompt_parts.append(f"\nBrand: {request.brand.name}")
            prompt_parts.append(f"Brand Voice: {request.brand.voice}")
            prompt_parts.append(f"Industry: {request.brand.industry}")
            
            if request.brand.values:
                prompt_parts.append(f"Brand Values: {', '.join(request.brand.values)}")
                
            if request.brand.keywords:
                prompt_parts.append(f"Brand Keywords: {', '.join(request.brand.keywords)}")
        
        # Style and audience
        style_instruction = self.style_prompts.get(request.style, "")
        audience_instruction = self.audience_prompts.get(request.target_audience, "")
        
        prompt_parts.append(f"\nStyle: {style_instruction}")
        prompt_parts.append(f"Target Audience: {audience_instruction}")
        
        # Content type specific instructions
        if request.content_type == ContentType.STORY:
            prompt_parts.append("Format for Instagram Story (shorter, more immediate)")
        elif request.content_type == ContentType.REEL:
            prompt_parts.append("Format for Instagram Reel (engaging, video-focused)")
        elif request.content_type == ContentType.CAROUSEL:
            prompt_parts.append("Format for carousel post (detailed, informative)")
        
        # Length and formatting requirements
        prompt_parts.append(f"\nRequirements:")
        prompt_parts.append(f"- Maximum {request.max_length} characters")
        
        if request.include_emojis:
            prompt_parts.append("- Include relevant emojis naturally throughout")
            
        if request.include_cta:
            prompt_parts.append("- Include a clear call-to-action")
            
        # Hashtag instructions
        if request.include_hashtags:
            hashtag_instruction = self.hashtag_strategies.get(request.hashtag_strategy, "")
            prompt_parts.append(f"- Include {request.hashtag_count} hashtags: {hashtag_instruction}")
        else:
            prompt_parts.append("- Do not include hashtags")
        
        # Final instruction
        prompt_parts.append("\nGenerate a caption that is authentic, engaging, and optimized for Instagram engagement.")
        
        return "\n".join(prompt_parts)

class HashtagGenerator:
    """Generate relevant hashtags for Instagram posts."""
    
    def __init__(self) -> Any:
        self.trending_hashtags = [
            "#trending", "#viral", "#explore", "#reels", "#instagram",
            "#love", "#instagood", "#photooftheday", "#beautiful", "#happy"
        ]
        
        self.niche_hashtags = {
            InstagramTarget.FASHION: ["#fashion", "#style", "#ootd", "#fashionista", "#styleinspo"],
            InstagramTarget.FOOD: ["#food", "#foodie", "#delicious", "#yummy", "#recipe"],
            InstagramTarget.TRAVEL: ["#travel", "#wanderlust", "#explore", "#adventure", "#vacation"],
            InstagramTarget.FITNESS: ["#fitness", "#workout", "#health", "#gym", "#motivation"],
            InstagramTarget.TECH: ["#tech", "#technology", "#innovation", "#digital", "#ai"]
        }
    
    def generate_hashtags(self, request: InstagramCaptionRequest, count: int) -> List[str]:
        """Generate relevant hashtags based on request."""
        hashtags = []
        
        # Add niche hashtags based on target audience
        if request.target_audience in self.niche_hashtags:
            hashtags.extend(self.niche_hashtags[request.target_audience][:count//2])
        
        # Add brand hashtags
        if request.brand and request.brand.keywords:
            brand_tags = [f"#{keyword.lower().replace(' ', '')}" for keyword in request.brand.keywords[:2]]
            hashtags.extend(brand_tags)
        
        # Add trending hashtags
        remaining = count - len(hashtags)
        if remaining > 0:
            hashtags.extend(self.trending_hashtags[:remaining])
        
        # Ensure we don't exceed the requested count
        return hashtags[:count]

class InstagramCaptionsService:
    """Main Instagram Captions generation service."""
    
    def __init__(self) -> Any:
        self.providers = {}
        self.prompt_builder = PromptBuilder()
        self.hashtag_generator = HashtagGenerator()
        self.captions_engine = InstagramCaptionsEngine()
        self.gmt_system = SimplifiedGMTSystem()
        self.initialize_providers()
    
    def initialize_providers(self) -> Any:
        """Initialize all AI providers."""
        # Initialize OpenAI
        if OPENAI_AVAILABLE:
            self.providers['openai'] = OpenAIProvider()
            
        # Initialize LangChain
        if LANGCHAIN_AVAILABLE:
            self.providers['langchain'] = LangChainProvider()
            
        # Initialize OpenRouter
        if REQUESTS_AVAILABLE:
            self.providers['openrouter'] = OpenRouterProvider()
        
        available_providers = [name for name, provider in self.providers.items() if provider.is_available()]
        logger.info(f"Initialized providers: {available_providers}")
    
    def get_best_provider(self, request: InstagramCaptionRequest) -> AIProvider:
        """Select the best available provider based on request preferences."""
        # Check user preferences
        if request.use_langchain and 'langchain' in self.providers and self.providers['langchain'].is_available():
            return self.providers['langchain']
            
        if request.use_openrouter and 'openrouter' in self.providers and self.providers['openrouter'].is_available():
            return self.providers['openrouter']
            
        if request.use_openai and 'openai' in self.providers and self.providers['openai'].is_available():
            return self.providers['openai']
        
        # Fallback to any available provider
        for provider in self.providers.values():
            if provider.is_available():
                return provider
        
        raise ValueError("No AI providers available")
    
    async def generate_single_caption(self, request: InstagramCaptionRequest, provider: AIProvider) -> CaptionVariation:
        """Generate a single caption variation with quality optimization."""
        start_time = time.perf_counter()
        
        try:
            # Create optimized prompt for better quality
            optimized_prompt = self.captions_engine.create_optimized_prompt(
                content_desc=request.content.description,
                style=request.style,
                audience=request.target_audience,
                brand_context=request.brand.__dict__ if request.brand else None
            )
            
            # Generate caption using AI provider with enhanced prompt
            caption = await provider.generate_caption(optimized_prompt)
            
            # Extract hashtags from caption if they exist
            lines = caption.split('\n')
            caption_text = ""
            hashtags = []
            
            for line in lines:
                if line.strip().startswith('#'):
                    # Extract hashtags from this line
                    tags = [tag.strip() for tag in line.split() if tag.startswith('#')]
                    hashtags.extend(tags)
                else:
                    caption_text += line + "\n"
            
            caption_text = caption_text.strip()
            
            # Generate enhanced hashtags using core engine
            if request.include_hashtags:
                content_keywords = self._extract_keywords(request.content.description)
                hashtags = self.captions_engine.generate_hashtags(
                    content_keywords=content_keywords,
                    audience=request.target_audience,
                    style=request.style,
                    strategy=request.hashtag_strategy,
                    count=request.hashtag_count
                )
            
            # Optimize caption quality using core engine
            optimized_caption, quality_metrics = await self.captions_engine.optimize_content(
                caption_text, request.style, request.target_audience
            )
            
            # Apply cultural adaptation if timezone provided
            timezone = getattr(request, 'timezone', 'UTC')
            if timezone != 'UTC':
                optimized_caption = self.gmt_system.adapt_content_culturally(
                    optimized_caption, timezone, request.style, request.target_audience
                )
            
            # Calculate metrics
            character_count = len(caption_text)
            word_count = len(caption_text.split())
            emoji_count = sum(1 for char in caption_text if ord(char) > 127)
            
            # Create variation with quality metrics
            variation = CaptionVariation(
                caption=optimized_caption,
                hashtags=hashtags if request.include_hashtags else [],
                character_count=len(optimized_caption),
                word_count=len(optimized_caption.split()),
                emoji_count=sum(1 for char in optimized_caption if ord(char) > 127),
                style_score=quality_metrics.hook_strength / 100,
                engagement_prediction=quality_metrics.engagement_potential / 100,
                readability_score=quality_metrics.readability / 100
            )
            
            generation_time = time.perf_counter() - start_time
            logger.info(f"Generated caption variation in {generation_time:.2f}s using {provider.name}")
            
            return variation
            
        except Exception as e:
            logger.error(f"Caption generation failed with {provider.name}: {e}")
            raise
    
    async def generate_captions(self, request: InstagramCaptionRequest) -> InstagramCaptionResponse:
        """Generate Instagram captions with multiple variations."""
        start_time = time.perf_counter()
        
        try:
            # Get best provider
            provider = self.get_best_provider(request)
            
            # Generate variations
            variations = []
            if request.generate_variations:
                # Generate multiple variations
                tasks = [
                    self.generate_single_caption(request, provider)
                    for _ in range(request.variation_count)
                ]
                variations = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                variations = [v for v in variations if isinstance(v, CaptionVariation)]
            else:
                # Generate single caption
                variation = await self.generate_single_caption(request, provider)
                variations = [variation]
            
            if not variations:
                raise ValueError("Failed to generate any caption variations")
            
            # Create response
            generation_time = time.perf_counter() - start_time
            
            # Find best variation (highest combined score)
            best_variation = max(variations, 
                               key=lambda v: v.style_score + v.engagement_prediction + v.readability_score)
            
            # Create timezone info (placeholder - would integrate with GMT system)
            timezone_info = TimeZoneInfo(
                timezone=request.target_timezone,
                current_time=datetime.now(timezone.utc),
                utc_offset="+00:00",
                local_hour=datetime.now().hour,
                optimal_posting_window=True,
                peak_engagement_time=False
            )
            
            # Create generation metrics
            metrics = GenerationMetrics(
                generation_time=generation_time,
                model_used=provider.name,
                provider_used=provider.name,
                token_count=sum(v.word_count for v in variations),
                cost_estimate=0.01 * len(variations),  # Rough estimate
                quality_score=sum(v.style_score for v in variations) / len(variations)
            )
            
            response = InstagramCaptionResponse(
                variations=variations,
                timezone_info=timezone_info,
                generation_metrics=metrics,
                best_variation_id=best_variation.id,
                posting_recommendations=[
                    "Post during peak engagement hours (6-9 PM)",
                    "Use trending hashtags for better reach",
                    "Engage with comments quickly after posting"
                ],
                optimization_suggestions=[
                    "Consider A/B testing different variations",
                    "Monitor engagement metrics",
                    "Adapt content based on audience response"
                ]
            )
            
            logger.info(f"Generated {len(variations)} caption variations in {generation_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from content description."""
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        # Get meaningful words
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Return top keywords
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(8)] 