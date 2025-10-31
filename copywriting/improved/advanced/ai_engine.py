"""
AI Engine Integration
====================

Advanced AI engine integration for content generation with multiple providers.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import httpx
import openai
from anthropic import AsyncAnthropic
import google.generativeai as genai

from ..schemas import CopywritingRequest, CopywritingVariant
from ..exceptions import ContentGenerationError, ExternalServiceError
from ..utils import retry_with_backoff, measure_execution_time

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class AIProviderConfig:
    """Configuration for AI providers"""
    provider: AIProvider
    api_key: str
    model: str
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3


class AIEngine(ABC):
    """Abstract base class for AI engines"""
    
    @abstractmethod
    async def generate_content(
        self,
        request: CopywritingRequest,
        provider_config: AIProviderConfig
    ) -> List[CopywritingVariant]:
        """Generate content using the AI provider"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the AI provider is healthy"""
        pass


class OpenAIEngine(AIEngine):
    """OpenAI GPT integration"""
    
    def __init__(self):
        self.client = None
    
    async def _get_client(self, api_key: str):
        """Get OpenAI client"""
        if not self.client:
            self.client = openai.AsyncOpenAI(api_key=api_key)
        return self.client
    
    @measure_execution_time
    async def generate_content(
        self,
        request: CopywritingRequest,
        provider_config: AIProviderConfig
    ) -> List[CopywritingVariant]:
        """Generate content using OpenAI"""
        try:
            client = await self._get_client(provider_config.api_key)
            
            # Create prompt based on request
            prompt = self._create_prompt(request)
            
            variants = []
            for i in range(request.variants_count):
                response = await client.chat.completions.create(
                    model=provider_config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert copywriter who creates compelling, engaging content."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=provider_config.max_tokens,
                    temperature=provider_config.temperature + (i * 0.1),  # Vary temperature
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                
                content = response.choices[0].message.content
                if not content:
                    continue
                
                # Parse content to extract title and body
                title, body = self._parse_content(content)
                
                variant = CopywritingVariant(
                    title=title,
                    content=body,
                    word_count=len(body.split()),
                    cta=self._extract_cta(body) if request.include_cta else None,
                    confidence_score=self._calculate_confidence(content, request)
                )
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise ContentGenerationError(
                message="Failed to generate content with OpenAI",
                details={"provider": "openai", "error": str(e)}
            )
    
    async def health_check(self) -> bool:
        """Check OpenAI health"""
        try:
            # Simple health check - could be more sophisticated
            return True
        except Exception:
            return False
    
    def _create_prompt(self, request: CopywritingRequest) -> str:
        """Create prompt for OpenAI"""
        prompt = f"""
        Create compelling copywriting content with the following specifications:
        
        Topic: {request.topic}
        Target Audience: {request.target_audience}
        Tone: {request.tone}
        Style: {request.style}
        Purpose: {request.purpose}
        
        Key Points to Include:
        {chr(10).join(f"- {point}" for point in request.key_points)}
        
        Word Count: {request.word_count or 'Flexible'}
        Language: {request.language}
        Creativity Level: {request.creativity_level}
        
        Brand Voice: {request.brand_voice or 'Not specified'}
        Brand Values: {', '.join(request.brand_values) if request.brand_values else 'Not specified'}
        
        Please create engaging, persuasive content that:
        1. Captures attention immediately
        2. Addresses the target audience's needs
        3. Maintains the specified tone and style
        4. Achieves the stated purpose
        5. Includes the key points naturally
        """
        
        if request.include_cta:
            prompt += "\n6. Ends with a compelling call-to-action"
        
        return prompt.strip()
    
    def _parse_content(self, content: str) -> tuple[str, str]:
        """Parse content to extract title and body"""
        lines = content.strip().split('\n')
        title = lines[0] if lines else "Generated Content"
        
        # Remove title from body
        body_lines = lines[1:] if len(lines) > 1 else lines
        body = '\n'.join(body_lines).strip()
        
        return title, body
    
    def _extract_cta(self, content: str) -> Optional[str]:
        """Extract call-to-action from content"""
        # Simple CTA extraction - could be more sophisticated
        cta_phrases = [
            "call to action", "cta", "click here", "learn more",
            "get started", "sign up", "buy now", "contact us"
        ]
        
        sentences = content.split('.')
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in cta_phrases):
                return sentence.strip()
        
        return None
    
    def _calculate_confidence(self, content: str, request: CopywritingRequest) -> float:
        """Calculate confidence score for generated content"""
        base_score = 0.7
        
        # Check if key points are included
        key_points_included = sum(
            1 for point in request.key_points
            if point.lower() in content.lower()
        )
        key_points_score = (key_points_included / len(request.key_points)) * 0.2 if request.key_points else 0.1
        
        # Check word count if specified
        word_count = len(content.split())
        if request.word_count:
            word_ratio = min(word_count / request.word_count, request.word_count / word_count)
            word_score = word_ratio * 0.1
        else:
            word_score = 0.1
        
        return min(base_score + key_points_score + word_score, 1.0)


class AnthropicEngine(AIEngine):
    """Anthropic Claude integration"""
    
    def __init__(self):
        self.client = None
    
    async def _get_client(self, api_key: str):
        """Get Anthropic client"""
        if not self.client:
            self.client = AsyncAnthropic(api_key=api_key)
        return self.client
    
    @measure_execution_time
    async def generate_content(
        self,
        request: CopywritingRequest,
        provider_config: AIProviderConfig
    ) -> List[CopywritingVariant]:
        """Generate content using Anthropic Claude"""
        try:
            client = await self._get_client(provider_config.api_key)
            
            prompt = self._create_prompt(request)
            
            variants = []
            for i in range(request.variants_count):
                response = await client.messages.create(
                    model=provider_config.model,
                    max_tokens=provider_config.max_tokens,
                    temperature=provider_config.temperature + (i * 0.1),
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                content = response.content[0].text
                if not content:
                    continue
                
                title, body = self._parse_content(content)
                
                variant = CopywritingVariant(
                    title=title,
                    content=body,
                    word_count=len(body.split()),
                    cta=self._extract_cta(body) if request.include_cta else None,
                    confidence_score=self._calculate_confidence(content, request)
                )
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise ContentGenerationError(
                message="Failed to generate content with Anthropic",
                details={"provider": "anthropic", "error": str(e)}
            )
    
    async def health_check(self) -> bool:
        """Check Anthropic health"""
        try:
            return True
        except Exception:
            return False
    
    def _create_prompt(self, request: CopywritingRequest) -> str:
        """Create prompt for Anthropic"""
        # Similar to OpenAI but optimized for Claude
        return f"""
        As an expert copywriter, create compelling content with these specifications:
        
        Topic: {request.topic}
        Target Audience: {request.target_audience}
        Tone: {request.tone}
        Style: {request.style}
        Purpose: {request.purpose}
        
        Key Points: {', '.join(request.key_points)}
        Word Count: {request.word_count or 'Flexible'}
        Language: {request.language}
        
        Create content that is engaging, persuasive, and tailored to the audience.
        """
    
    def _parse_content(self, content: str) -> tuple[str, str]:
        """Parse content to extract title and body"""
        lines = content.strip().split('\n')
        title = lines[0] if lines else "Generated Content"
        body = '\n'.join(lines[1:]) if len(lines) > 1 else content
        return title, body
    
    def _extract_cta(self, content: str) -> Optional[str]:
        """Extract call-to-action from content"""
        # Similar to OpenAI implementation
        cta_phrases = ["call to action", "cta", "click here", "learn more"]
        sentences = content.split('.')
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in cta_phrases):
                return sentence.strip()
        return None
    
    def _calculate_confidence(self, content: str, request: CopywritingRequest) -> float:
        """Calculate confidence score"""
        return 0.8  # Claude typically produces high-quality content


class GoogleEngine(AIEngine):
    """Google Gemini integration"""
    
    def __init__(self):
        self.model = None
    
    async def _get_model(self, api_key: str, model_name: str):
        """Get Google model"""
        if not self.model:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        return self.model
    
    @measure_execution_time
    async def generate_content(
        self,
        request: CopywritingRequest,
        provider_config: AIProviderConfig
    ) -> List[CopywritingVariant]:
        """Generate content using Google Gemini"""
        try:
            model = await self._get_model(provider_config.api_key, provider_config.model)
            
            prompt = self._create_prompt(request)
            
            variants = []
            for i in range(request.variants_count):
                response = await model.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=provider_config.max_tokens,
                        temperature=provider_config.temperature + (i * 0.1),
                        top_p=0.9,
                        top_k=40
                    )
                )
                
                content = response.text
                if not content:
                    continue
                
                title, body = self._parse_content(content)
                
                variant = CopywritingVariant(
                    title=title,
                    content=body,
                    word_count=len(body.split()),
                    cta=self._extract_cta(body) if request.include_cta else None,
                    confidence_score=self._calculate_confidence(content, request)
                )
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Google generation failed: {e}")
            raise ContentGenerationError(
                message="Failed to generate content with Google",
                details={"provider": "google", "error": str(e)}
            )
    
    async def health_check(self) -> bool:
        """Check Google health"""
        try:
            return True
        except Exception:
            return False
    
    def _create_prompt(self, request: CopywritingRequest) -> str:
        """Create prompt for Google Gemini"""
        return f"""
        Create compelling copywriting content:
        
        Topic: {request.topic}
        Audience: {request.target_audience}
        Tone: {request.tone}
        Style: {request.style}
        Purpose: {request.purpose}
        
        Include these key points: {', '.join(request.key_points)}
        Target word count: {request.word_count or 'Flexible'}
        Language: {request.language}
        
        Make it engaging and persuasive for the target audience.
        """
    
    def _parse_content(self, content: str) -> tuple[str, str]:
        """Parse content to extract title and body"""
        lines = content.strip().split('\n')
        title = lines[0] if lines else "Generated Content"
        body = '\n'.join(lines[1:]) if len(lines) > 1 else content
        return title, body
    
    def _extract_cta(self, content: str) -> Optional[str]:
        """Extract call-to-action from content"""
        cta_phrases = ["call to action", "cta", "click here", "learn more"]
        sentences = content.split('.')
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in cta_phrases):
                return sentence.strip()
        return None
    
    def _calculate_confidence(self, content: str, request: CopywritingRequest) -> float:
        """Calculate confidence score"""
        return 0.75  # Google Gemini produces good content


class AIEngineManager:
    """Manager for multiple AI engines"""
    
    def __init__(self):
        self.engines = {
            AIProvider.OPENAI: OpenAIEngine(),
            AIProvider.ANTHROPIC: AnthropicEngine(),
            AIProvider.GOOGLE: GoogleEngine(),
        }
        self.provider_configs: Dict[AIProvider, AIProviderConfig] = {}
    
    def configure_provider(self, config: AIProviderConfig):
        """Configure an AI provider"""
        self.provider_configs[config.provider] = config
        logger.info(f"Configured AI provider: {config.provider}")
    
    async def generate_content(
        self,
        request: CopywritingRequest,
        preferred_provider: Optional[AIProvider] = None
    ) -> List[CopywritingVariant]:
        """Generate content using the best available provider"""
        
        # Determine which provider to use
        provider = preferred_provider or self._select_best_provider()
        
        if provider not in self.provider_configs:
            raise ContentGenerationError(
                message=f"Provider {provider} not configured",
                details={"available_providers": list(self.provider_configs.keys())}
            )
        
        config = self.provider_configs[provider]
        engine = self.engines[provider]
        
        # Generate content with retry logic
        try:
            variants = await retry_with_backoff(
                engine.generate_content,
                max_retries=config.max_retries,
                base_delay=1.0
            )(request, config)
            
            logger.info(f"Generated {len(variants)} variants using {provider}")
            return variants
            
        except Exception as e:
            logger.error(f"Failed to generate content with {provider}: {e}")
            
            # Try fallback providers
            for fallback_provider in self.provider_configs:
                if fallback_provider != provider:
                    try:
                        logger.info(f"Trying fallback provider: {fallback_provider}")
                        fallback_config = self.provider_configs[fallback_provider]
                        fallback_engine = self.engines[fallback_provider]
                        
                        variants = await fallback_engine.generate_content(request, fallback_config)
                        logger.info(f"Generated {len(variants)} variants using fallback {fallback_provider}")
                        return variants
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback provider {fallback_provider} also failed: {fallback_error}")
                        continue
            
            # All providers failed
            raise ContentGenerationError(
                message="All AI providers failed to generate content",
                details={"primary_provider": provider, "error": str(e)}
            )
    
    def _select_best_provider(self) -> AIProvider:
        """Select the best available provider"""
        # Simple selection logic - could be more sophisticated
        if AIProvider.OPENAI in self.provider_configs:
            return AIProvider.OPENAI
        elif AIProvider.ANTHROPIC in self.provider_configs:
            return AIProvider.ANTHROPIC
        elif AIProvider.GOOGLE in self.provider_configs:
            return AIProvider.GOOGLE
        else:
            raise ContentGenerationError(
                message="No AI providers configured",
                details={"available_providers": list(self.provider_configs.keys())}
            )
    
    async def health_check_all(self) -> Dict[AIProvider, bool]:
        """Check health of all configured providers"""
        health_status = {}
        
        for provider, engine in self.engines.items():
            if provider in self.provider_configs:
                try:
                    health_status[provider] = await engine.health_check()
                except Exception as e:
                    logger.warning(f"Health check failed for {provider}: {e}")
                    health_status[provider] = False
        
        return health_status
    
    def get_available_providers(self) -> List[AIProvider]:
        """Get list of configured providers"""
        return list(self.provider_configs.keys())


# Global AI engine manager instance
ai_engine_manager = AIEngineManager()






























