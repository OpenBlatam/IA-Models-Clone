"""
AI Service
==========

Advanced AI service for multi-provider AI integration and workflow automation.
"""

from __future__ import annotations
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import openai
from anthropic import Anthropic
import google.generativeai as genai
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution, retry, measure_performance


logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """AI provider enumeration"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"


class AIModelType(str, Enum):
    """AI model type enumeration"""
    TEXT_GENERATION = "text_generation"
    TEXT_ANALYSIS = "text_analysis"
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_EXTRACTION = "entity_extraction"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"


class AIQualityLevel(str, Enum):
    """AI quality level enumeration"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    MAXIMUM = "maximum"


@dataclass
class AIRequest:
    """AI request representation"""
    id: str
    provider: AIProvider
    model: str
    model_type: AIModelType
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_level: AIQualityLevel = AIQualityLevel.BALANCED
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=DateTimeHelpers.now_utc)


@dataclass
class AIResponse:
    """AI response representation"""
    id: str
    request_id: str
    provider: AIProvider
    model: str
    content: str
    tokens_used: int
    processing_time: float
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=DateTimeHelpers.now_utc)


@dataclass
class AIModel:
    """AI model representation"""
    name: str
    provider: AIProvider
    model_type: AIModelType
    max_tokens: int
    cost_per_token: float
    quality_level: AIQualityLevel
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = True


class AIService:
    """Advanced AI service with multi-provider support"""
    
    def __init__(self):
        self.providers: Dict[AIProvider, Any] = {}
        self.models: Dict[str, AIModel] = {}
        self.requests: List[AIRequest] = []
        self.responses: List[AIResponse] = []
        self.is_running = False
        self._initialize_providers()
        self._load_models()
    
    def _initialize_providers(self):
        """Initialize AI providers"""
        try:
            # OpenAI
            openai.api_key = "your-openai-api-key"  # In real implementation, use env vars
            self.providers[AIProvider.OPENAI] = openai
            
            # Anthropic
            self.providers[AIProvider.ANTHROPIC] = Anthropic(api_key="your-anthropic-api-key")
            
            # Google
            genai.configure(api_key="your-google-api-key")
            self.providers[AIProvider.GOOGLE] = genai
            
            # Azure
            self.providers[AIProvider.AZURE] = TextAnalyticsClient(
                endpoint="your-azure-endpoint",
                credential=AzureKeyCredential("your-azure-key")
            )
            
            logger.info("AI providers initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize AI providers: {e}")
    
    def _load_models(self):
        """Load available AI models"""
        # OpenAI models
        self.models["gpt-4"] = AIModel(
            name="gpt-4",
            provider=AIProvider.OPENAI,
            model_type=AIModelType.TEXT_GENERATION,
            max_tokens=8192,
            cost_per_token=0.00003,
            quality_level=AIQualityLevel.HIGH_QUALITY,
            capabilities=["text_generation", "text_analysis", "code_generation"]
        )
        
        self.models["gpt-3.5-turbo"] = AIModel(
            name="gpt-3.5-turbo",
            provider=AIProvider.OPENAI,
            model_type=AIModelType.TEXT_GENERATION,
            max_tokens=4096,
            cost_per_token=0.000002,
            quality_level=AIQualityLevel.BALANCED,
            capabilities=["text_generation", "text_analysis"]
        )
        
        # Anthropic models
        self.models["claude-3-opus"] = AIModel(
            name="claude-3-opus",
            provider=AIProvider.ANTHROPIC,
            model_type=AIModelType.TEXT_GENERATION,
            max_tokens=200000,
            cost_per_token=0.000015,
            quality_level=AIQualityLevel.MAXIMUM,
            capabilities=["text_generation", "text_analysis", "code_generation"]
        )
        
        self.models["claude-3-sonnet"] = AIModel(
            name="claude-3-sonnet",
            provider=AIProvider.ANTHROPIC,
            model_type=AIModelType.TEXT_GENERATION,
            max_tokens=200000,
            cost_per_token=0.000003,
            quality_level=AIQualityLevel.HIGH_QUALITY,
            capabilities=["text_generation", "text_analysis"]
        )
        
        # Google models
        self.models["gemini-pro"] = AIModel(
            name="gemini-pro",
            provider=AIProvider.GOOGLE,
            model_type=AIModelType.TEXT_GENERATION,
            max_tokens=32768,
            cost_per_token=0.000001,
            quality_level=AIQualityLevel.BALANCED,
            capabilities=["text_generation", "text_analysis", "translation"]
        )
        
        logger.info(f"Loaded {len(self.models)} AI models")
    
    async def start(self):
        """Start the AI service"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("AI service started")
    
    async def stop(self):
        """Stop the AI service"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("AI service stopped")
    
    @measure_performance
    async def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[AIProvider] = None,
        model_type: AIModelType = AIModelType.TEXT_GENERATION,
        quality_level: AIQualityLevel = AIQualityLevel.BALANCED,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AIResponse:
        """Generate content using AI"""
        # Select model
        selected_model = self._select_model(model, provider, model_type, quality_level)
        
        # Create request
        request = AIRequest(
            id=f"req_{int(DateTimeHelpers.now_utc().timestamp())}_{len(self.requests)}",
            provider=selected_model.provider,
            model=selected_model.name,
            model_type=model_type,
            prompt=prompt,
            parameters={
                "max_tokens": max_tokens or selected_model.max_tokens,
                "temperature": temperature
            },
            quality_level=quality_level,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.requests.append(request)
        
        # Generate content
        start_time = DateTimeHelpers.now_utc()
        
        try:
            content, tokens_used = await self._generate_with_provider(request)
            
            processing_time = (DateTimeHelpers.now_utc() - start_time).total_seconds()
            
            # Create response
            response = AIResponse(
                id=f"resp_{int(DateTimeHelpers.now_utc().timestamp())}_{len(self.responses)}",
                request_id=request.id,
                provider=selected_model.provider,
                model=selected_model.name,
                content=content,
                tokens_used=tokens_used,
                processing_time=processing_time,
                quality_score=self._calculate_quality_score(content, selected_model),
                confidence_score=self._calculate_confidence_score(content),
                metadata=request.metadata
            )
            
            self.responses.append(response)
            
            logger.info(f"Generated content using {selected_model.provider.value}:{selected_model.name}")
            
            return response
        
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            raise
    
    async def _generate_with_provider(self, request: AIRequest) -> Tuple[str, int]:
        """Generate content with specific provider"""
        if request.provider == AIProvider.OPENAI:
            return await self._generate_openai(request)
        elif request.provider == AIProvider.ANTHROPIC:
            return await self._generate_anthropic(request)
        elif request.provider == AIProvider.GOOGLE:
            return await self._generate_google(request)
        elif request.provider == AIProvider.AZURE:
            return await self._generate_azure(request)
        else:
            raise ValueError(f"Unsupported provider: {request.provider}")
    
    async def _generate_openai(self, request: AIRequest) -> Tuple[str, int]:
        """Generate content using OpenAI"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.parameters.get("max_tokens", 1000),
                temperature=request.parameters.get("temperature", 0.7)
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return content, tokens_used
        
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def _generate_anthropic(self, request: AIRequest) -> Tuple[str, int]:
        """Generate content using Anthropic"""
        try:
            client = self.providers[AIProvider.ANTHROPIC]
            
            response = await client.messages.create(
                model=request.model,
                max_tokens=request.parameters.get("max_tokens", 1000),
                temperature=request.parameters.get("temperature", 0.7),
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return content, tokens_used
        
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def _generate_google(self, request: AIRequest) -> Tuple[str, int]:
        """Generate content using Google"""
        try:
            model = self.providers[AIProvider.GOOGLE].GenerativeModel(request.model)
            
            response = await model.generate_content_async(
                request.prompt,
                generation_config={
                    "max_output_tokens": request.parameters.get("max_tokens", 1000),
                    "temperature": request.parameters.get("temperature", 0.7)
                }
            )
            
            content = response.text
            # Google doesn't provide token count in the same way
            tokens_used = len(content.split()) * 1.3  # Rough estimate
            
            return content, int(tokens_used)
        
        except Exception as e:
            logger.error(f"Google generation failed: {e}")
            raise
    
    async def _generate_azure(self, request: AIRequest) -> Tuple[str, int]:
        """Generate content using Azure"""
        try:
            # Azure Text Analytics for analysis tasks
            if request.model_type in [AIModelType.TEXT_ANALYSIS, AIModelType.SENTIMENT_ANALYSIS]:
                client = self.providers[AIProvider.AZURE]
                
                if request.model_type == AIModelType.SENTIMENT_ANALYSIS:
                    response = await client.analyze_sentiment([request.prompt])
                    content = str(response[0].sentiment)
                else:
                    response = await client.analyze([request.prompt])
                    content = str(response[0])
                
                tokens_used = len(request.prompt.split())
                
                return content, tokens_used
            else:
                raise ValueError(f"Azure doesn't support {request.model_type.value}")
        
        except Exception as e:
            logger.error(f"Azure generation failed: {e}")
            raise
    
    def _select_model(
        self,
        model: Optional[str],
        provider: Optional[AIProvider],
        model_type: AIModelType,
        quality_level: AIQualityLevel
    ) -> AIModel:
        """Select appropriate model based on criteria"""
        if model and model in self.models:
            return self.models[model]
        
        # Filter models by criteria
        candidates = []
        for model_obj in self.models.values():
            if not model_obj.is_active:
                continue
            
            if provider and model_obj.provider != provider:
                continue
            
            if model_type not in model_obj.capabilities:
                continue
            
            if model_obj.quality_level == quality_level:
                candidates.append(model_obj)
        
        if not candidates:
            # Fall back to any model that supports the type
            for model_obj in self.models.values():
                if model_obj.is_active and model_type in model_obj.capabilities:
                    candidates.append(model_obj)
        
        if not candidates:
            raise ValueError(f"No suitable model found for {model_type.value}")
        
        # Select best model based on quality and cost
        best_model = min(candidates, key=lambda m: (m.cost_per_token, -m.max_tokens))
        
        return best_model
    
    def _calculate_quality_score(self, content: str, model: AIModel) -> float:
        """Calculate quality score for generated content"""
        # Simple quality metrics
        length_score = min(len(content) / 100, 1.0)  # Prefer longer content
        model_quality = {
            AIQualityLevel.FAST: 0.6,
            AIQualityLevel.BALANCED: 0.8,
            AIQualityLevel.HIGH_QUALITY: 0.9,
            AIQualityLevel.MAXIMUM: 1.0
        }.get(model.quality_level, 0.8)
        
        return (length_score + model_quality) / 2
    
    def _calculate_confidence_score(self, content: str) -> float:
        """Calculate confidence score for generated content"""
        # Simple confidence metrics based on content characteristics
        if not content:
            return 0.0
        
        # Check for common error patterns
        error_patterns = ["error", "failed", "unable", "cannot", "sorry"]
        error_count = sum(1 for pattern in error_patterns if pattern.lower() in content.lower())
        
        # Check for completeness indicators
        completeness_indicators = [".", "!", "?", ":", ";"]
        completeness_score = min(len([c for c in content if c in completeness_indicators]) / 5, 1.0)
        
        # Calculate confidence
        confidence = max(0.0, 1.0 - (error_count * 0.2) + (completeness_score * 0.3))
        
        return min(confidence, 1.0)
    
    async def analyze_text(
        self,
        text: str,
        analysis_type: AIModelType = AIModelType.TEXT_ANALYSIS,
        provider: Optional[AIProvider] = None,
        user_id: Optional[str] = None
    ) -> AIResponse:
        """Analyze text using AI"""
        prompt = self._create_analysis_prompt(text, analysis_type)
        
        return await self.generate_content(
            prompt=prompt,
            model_type=analysis_type,
            provider=provider,
            user_id=user_id
        )
    
    def _create_analysis_prompt(self, text: str, analysis_type: AIModelType) -> str:
        """Create analysis prompt based on type"""
        prompts = {
            AIModelType.TEXT_SUMMARIZATION: f"Please summarize the following text:\n\n{text}",
            AIModelType.SENTIMENT_ANALYSIS: f"Analyze the sentiment of the following text:\n\n{text}",
            AIModelType.ENTITY_EXTRACTION: f"Extract entities from the following text:\n\n{text}",
            AIModelType.TEXT_CLASSIFICATION: f"Classify the following text:\n\n{text}",
            AIModelType.TRANSLATION: f"Translate the following text to English:\n\n{text}"
        }
        
        return prompts.get(analysis_type, f"Analyze the following text:\n\n{text}")
    
    async def batch_generate(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        provider: Optional[AIProvider] = None,
        max_concurrent: int = 5
    ) -> List[AIResponse]:
        """Generate content for multiple prompts in batch"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt: str) -> AIResponse:
            async with semaphore:
                return await self.generate_content(
                    prompt=prompt,
                    model=model,
                    provider=provider
                )
        
        tasks = [generate_single(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = [r for r in responses if isinstance(r, AIResponse)]
        
        logger.info(f"Batch generation completed: {len(valid_responses)}/{len(prompts)} successful")
        
        return valid_responses
    
    def get_available_models(self, provider: Optional[AIProvider] = None) -> List[AIModel]:
        """Get available models"""
        models = list(self.models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        return [m for m in models if m.is_active]
    
    def get_model_info(self, model_name: str) -> Optional[AIModel]:
        """Get model information"""
        return self.models.get(model_name)
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        total_requests = len(self.requests)
        total_responses = len(self.responses)
        
        # Provider usage
        provider_usage = {}
        for response in self.responses:
            provider = response.provider.value
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        # Model usage
        model_usage = {}
        for response in self.responses:
            model = response.model
            model_usage[model] = model_usage.get(model, 0) + 1
        
        # Token usage
        total_tokens = sum(r.tokens_used for r in self.responses)
        avg_tokens = total_tokens / total_responses if total_responses > 0 else 0
        
        # Performance metrics
        avg_processing_time = sum(r.processing_time for r in self.responses) / total_responses if total_responses > 0 else 0
        
        return {
            "total_requests": total_requests,
            "total_responses": total_responses,
            "success_rate": total_responses / total_requests if total_requests > 0 else 0,
            "provider_usage": provider_usage,
            "model_usage": model_usage,
            "total_tokens": total_tokens,
            "average_tokens": avg_tokens,
            "average_processing_time": avg_processing_time,
            "available_models": len(self.models),
            "active_models": len([m for m in self.models.values() if m.is_active]),
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }


# Global AI service
ai_service = AIService()


# Utility functions
async def start_ai_service():
    """Start the AI service"""
    await ai_service.start()


async def stop_ai_service():
    """Stop the AI service"""
    await ai_service.stop()


async def generate_content(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[AIProvider] = None,
    model_type: AIModelType = AIModelType.TEXT_GENERATION,
    quality_level: AIQualityLevel = AIQualityLevel.BALANCED,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> AIResponse:
    """Generate content using AI"""
    return await ai_service.generate_content(
        prompt=prompt,
        model=model,
        provider=provider,
        model_type=model_type,
        quality_level=quality_level,
        max_tokens=max_tokens,
        temperature=temperature,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata
    )


async def analyze_text(
    text: str,
    analysis_type: AIModelType = AIModelType.TEXT_ANALYSIS,
    provider: Optional[AIProvider] = None,
    user_id: Optional[str] = None
) -> AIResponse:
    """Analyze text using AI"""
    return await ai_service.analyze_text(text, analysis_type, provider, user_id)


async def batch_generate(
    prompts: List[str],
    model: Optional[str] = None,
    provider: Optional[AIProvider] = None,
    max_concurrent: int = 5
) -> List[AIResponse]:
    """Generate content for multiple prompts in batch"""
    return await ai_service.batch_generate(prompts, model, provider, max_concurrent)


def get_available_models(provider: Optional[AIProvider] = None) -> List[AIModel]:
    """Get available models"""
    return ai_service.get_available_models(provider)


def get_model_info(model_name: str) -> Optional[AIModel]:
    """Get model information"""
    return ai_service.get_model_info(model_name)


def get_usage_statistics() -> Dict[str, Any]:
    """Get usage statistics"""
    return ai_service.get_usage_statistics()


# Common AI operations
async def summarize_text(text: str, max_length: int = 200, user_id: Optional[str] = None) -> str:
    """Summarize text"""
    prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
    response = await generate_content(prompt, model_type=AIModelType.TEXT_SUMMARIZATION, user_id=user_id)
    return response.content


async def extract_entities(text: str, user_id: Optional[str] = None) -> str:
    """Extract entities from text"""
    response = await analyze_text(text, AIModelType.ENTITY_EXTRACTION, user_id=user_id)
    return response.content


async def analyze_sentiment(text: str, user_id: Optional[str] = None) -> str:
    """Analyze sentiment of text"""
    response = await analyze_text(text, AIModelType.SENTIMENT_ANALYSIS, user_id=user_id)
    return response.content


async def translate_text(text: str, target_language: str = "English", user_id: Optional[str] = None) -> str:
    """Translate text"""
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    response = await generate_content(prompt, model_type=AIModelType.TRANSLATION, user_id=user_id)
    return response.content


async def generate_code(description: str, language: str = "python", user_id: Optional[str] = None) -> str:
    """Generate code from description"""
    prompt = f"Generate {language} code for the following description:\n\n{description}"
    response = await generate_content(prompt, model_type=AIModelType.CODE_GENERATION, user_id=user_id)
    return response.content




