"""
Refactored LangChain Email Service

A modern, highly optimized LangChain service with:
- Advanced AI/ML Integration
- Performance Optimization with Cutting-edge Libraries
- Modern Python Patterns and Best Practices
- Comprehensive Error Handling and Resilience
- Intelligent Caching and Memory Management
"""

import asyncio
import logging
import time
import gc
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps, lru_cache
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, 
    AsyncGenerator, Protocol, runtime_checkable
)
from uuid import UUID
import weakref

# High-performance libraries
import orjson
import msgspec
import structlog
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)
import pybreaker
from cachetools import TTLCache, LRUCache
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    pipeline, TextGenerationPipeline
)
import openai
from openai import AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic
import cohere
from cohere import AsyncClient as AsyncCohere
import replicate
from huggingface_hub import AsyncInferenceClient

# LangChain imports
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers import VectorStoreRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

# Models
from ..models.sequence import EmailSequence, SequenceStep, SequenceTrigger
from ..models.subscriber import Subscriber, SubscriberSegment
from ..models.template import EmailTemplate, TemplateVariable

# Configure structured logging
logger = structlog.get_logger(__name__)

# Constants
MAX_TOKENS = 4096
TEMPERATURE = 0.7
CACHE_TTL = 3600  # 1 hour
CACHE_SIZE = 1000
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30


class ModelProvider(Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    LOCAL = "local"


class ContentType(Enum):
    """Content types for personalization"""
    EMAIL = "email"
    SUBJECT = "subject"
    TEMPLATE = "template"
    SEQUENCE = "sequence"
    CAMPAIGN = "campaign"


@dataclass
class LangChainConfig:
    """Configuration for LangChain service"""
    provider: ModelProvider = ModelProvider.OPENAI
    api_key: Optional[str] = None
    model_name: str = "gpt-4"
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    enable_caching: bool = True
    enable_streaming: bool = True
    enable_embeddings: bool = True
    enable_vector_store: bool = True
    cache_ttl: int = CACHE_TTL
    cache_size: int = CACHE_SIZE
    max_retries: int = MAX_RETRIES
    timeout_seconds: int = TIMEOUT_SECONDS


@dataclass
class PersonalizationContext:
    """Context for content personalization"""
    subscriber: Subscriber
    sequence: EmailSequence
    step: Optional[SequenceStep] = None
    template: Optional[EmailTemplate] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of content generation"""
    content: str
    metadata: Dict[str, Any]
    tokens_used: int
    model_used: str
    generation_time: float
    success: bool = True
    error: Optional[str] = None


class ModelManager:
    """Advanced model management with multiple providers"""
    
    def __init__(self, config: LangChainConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.embeddings: Dict[str, Any] = {}
        self.vector_stores: Dict[str, Any] = {}
        self.cache = TTLCache(maxsize=config.cache_size, ttl=config.cache_ttl)
        
        # Circuit breakers for different providers
        self.circuit_breakers = {
            ModelProvider.OPENAI: pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60),
            ModelProvider.ANTHROPIC: pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60),
            ModelProvider.COHERE: pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60),
            ModelProvider.HUGGINGFACE: pybreaker.CircuitBreaker(fail_max=3, reset_timeout=30),
            ModelProvider.REPLICATE: pybreaker.CircuitBreaker(fail_max=3, reset_timeout=30),
        }
    
    async def initialize(self) -> None:
        """Initialize models and embeddings"""
        try:
            if self.config.provider == ModelProvider.OPENAI:
                await self._initialize_openai()
            elif self.config.provider == ModelProvider.ANTHROPIC:
                await self._initialize_anthropic()
            elif self.config.provider == ModelProvider.COHERE:
                await self._initialize_cohere()
            elif self.config.provider == ModelProvider.HUGGINGFACE:
                await self._initialize_huggingface()
            elif self.config.provider == ModelProvider.REPLICATE:
                await self._initialize_replicate()
            
            # Initialize embeddings if enabled
            if self.config.enable_embeddings:
                await self._initialize_embeddings()
            
            logger.info("Model manager initialized successfully", provider=self.config.provider.value)
            
        except Exception as e:
            logger.error("Failed to initialize model manager", error=str(e))
            raise
    
    async def _initialize_openai(self) -> None:
        """Initialize OpenAI models"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.config.api_key
        
        # Initialize chat model
        self.models["chat"] = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=self.config.enable_streaming
        )
        
        # Initialize embeddings
        if self.config.enable_embeddings:
            self.embeddings["openai"] = OpenAIEmbeddings(
                openai_api_key=self.config.api_key
            )
    
    async def _initialize_anthropic(self) -> None:
        """Initialize Anthropic models"""
        if not self.config.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Initialize chat model
        self.models["chat"] = ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens_to_sample=self.config.max_tokens,
            anthropic_api_key=self.config.api_key
        )
    
    async def _initialize_cohere(self) -> None:
        """Initialize Cohere models"""
        if not self.config.api_key:
            raise ValueError("Cohere API key is required")
        
        # Initialize client
        self.models["client"] = AsyncCohere(api_token=self.config.api_key)
    
    async def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace models"""
        # Initialize tokenizer and model
        model_name = self.config.model_name or "gpt2"
        
        try:
            self.models["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            self.models["model"] = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Initialize text generation pipeline
            self.models["pipeline"] = pipeline(
                "text-generation",
                model=self.models["model"],
                tokenizer=self.models["tokenizer"],
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model {model_name}: {e}")
            # Fallback to a smaller model
            fallback_model = "distilgpt2"
            self.models["tokenizer"] = AutoTokenizer.from_pretrained(fallback_model)
            self.models["model"] = AutoModelForCausalLM.from_pretrained(fallback_model)
            self.models["pipeline"] = pipeline(
                "text-generation",
                model=self.models["model"],
                tokenizer=self.models["tokenizer"]
            )
    
    async def _initialize_replicate(self) -> None:
        """Initialize Replicate models"""
        if not self.config.api_key:
            raise ValueError("Replicate API key is required")
        
        # Set API token
        import os
        os.environ["REPLICATE_API_TOKEN"] = self.config.api_key
    
    async def _initialize_embeddings(self) -> None:
        """Initialize embeddings"""
        try:
            # Initialize HuggingFace embeddings as fallback
            self.embeddings["huggingface"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
    
    @lru_cache(maxsize=100)
    def get_prompt_template(self, template_name: str) -> PromptTemplate:
        """Get cached prompt template"""
        templates = {
            "email_generation": PromptTemplate(
                input_variables=["audience", "goals", "tone", "context"],
                template="""
                Generate an engaging email for the following context:
                
                Target Audience: {audience}
                Goals: {goals}
                Tone: {tone}
                Context: {context}
                
                Please create a compelling email that:
                1. Captures attention with a strong subject line
                2. Addresses the audience's needs and interests
                3. Maintains the specified tone throughout
                4. Includes a clear call-to-action
                5. Is optimized for email deliverability
                
                Email:
                """
            ),
            "subject_line": PromptTemplate(
                input_variables=["email_content", "audience", "goals"],
                template="""
                Based on this email content, generate 5 compelling subject lines:
                
                Email Content: {email_content}
                Target Audience: {audience}
                Goals: {goals}
                
                Generate subject lines that are:
                1. Under 50 characters
                2. Engaging and curiosity-driven
                3. Relevant to the content
                4. Optimized for open rates
                5. A/B test ready
                
                Subject Lines:
                """
            ),
            "personalization": PromptTemplate(
                input_variables=["content", "subscriber_data", "variables"],
                template="""
                Personalize this email content for the subscriber:
                
                Original Content: {content}
                Subscriber Data: {subscriber_data}
                Personalization Variables: {variables}
                
                Please personalize the content by:
                1. Using the subscriber's name appropriately
                2. Incorporating relevant subscriber data
                3. Maintaining the original tone and message
                4. Ensuring natural language flow
                5. Preserving all formatting and links
                
                Personalized Content:
                """
            ),
            "sequence_creation": PromptTemplate(
                input_variables=["name", "audience", "goals", "tone", "steps_count"],
                template="""
                Create an email sequence with the following specifications:
                
                Sequence Name: {name}
                Target Audience: {audience}
                Goals: {goals}
                Tone: {tone}
                Number of Steps: {steps_count}
                
                Please create a sequence that:
                1. Builds a relationship with the audience
                2. Gradually moves them toward the goal
                3. Includes appropriate delays between emails
                4. Uses varied content types (welcome, nurture, conversion)
                5. Has clear progression and flow
                
                Sequence Structure:
                """
            )
        }
        
        return templates.get(template_name, templates["email_generation"])
    
    async def generate_content(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> GenerationResult:
        """Generate content using the configured model"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"content:{hash(prompt + str(context))}"
            if use_cache and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                logger.info("Content retrieved from cache", cache_key=cache_key)
                return cached_result
            
            # Generate content based on provider
            if self.config.provider == ModelProvider.OPENAI:
                result = await self._generate_openai(prompt, context)
            elif self.config.provider == ModelProvider.ANTHROPIC:
                result = await self._generate_anthropic(prompt, context)
            elif self.config.provider == ModelProvider.COHERE:
                result = await self._generate_cohere(prompt, context)
            elif self.config.provider == ModelProvider.HUGGINGFACE:
                result = await self._generate_huggingface(prompt, context)
            elif self.config.provider == ModelProvider.REPLICATE:
                result = await self._generate_replicate(prompt, context)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            # Cache result
            if use_cache:
                self.cache[cache_key] = result
            
            generation_time = time.time() - start_time
            result.generation_time = generation_time
            
            logger.info("Content generated successfully",
                       provider=self.config.provider.value,
                       generation_time=generation_time,
                       tokens_used=result.tokens_used)
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            
            logger.error("Content generation failed",
                        provider=self.config.provider.value,
                        error=str(e),
                        generation_time=generation_time)
            
            return GenerationResult(
                content="",
                metadata={},
                tokens_used=0,
                model_used=self.config.model_name,
                generation_time=generation_time,
                success=False,
                error=str(e)
            )
    
    async def _generate_openai(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Generate content using OpenAI"""
        try:
            client = AsyncOpenAI(api_key=self.config.api_key)
            
            messages = [
                {"role": "system", "content": "You are an expert email marketing specialist."},
                {"role": "user", "content": prompt}
            ]
            
            response = await client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return GenerationResult(
                content=content,
                metadata={"model": self.config.model_name, "provider": "openai"},
                tokens_used=tokens_used,
                model_used=self.config.model_name
            )
            
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {e}")
    
    async def _generate_anthropic(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Generate content using Anthropic"""
        try:
            client = AsyncAnthropic(api_key=self.config.api_key)
            
            response = await client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return GenerationResult(
                content=content,
                metadata={"model": self.config.model_name, "provider": "anthropic"},
                tokens_used=tokens_used,
                model_used=self.config.model_name
            )
            
        except Exception as e:
            raise Exception(f"Anthropic generation failed: {e}")
    
    async def _generate_cohere(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Generate content using Cohere"""
        try:
            client = self.models["client"]
            
            response = await client.generate(
                model=self.config.model_name,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                num_generations=1
            )
            
            content = response.generations[0].text
            tokens_used = response.meta.billed_units.input_tokens + response.meta.billed_units.output_tokens
            
            return GenerationResult(
                content=content,
                metadata={"model": self.config.model_name, "provider": "cohere"},
                tokens_used=tokens_used,
                model_used=self.config.model_name
            )
            
        except Exception as e:
            raise Exception(f"Cohere generation failed: {e}")
    
    async def _generate_huggingface(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Generate content using HuggingFace"""
        try:
            pipeline = self.models["pipeline"]
            
            # Generate text
            result = pipeline(
                prompt,
                max_length=len(prompt.split()) + self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            
            content = result[0]["generated_text"]
            # Remove the original prompt
            content = content[len(prompt):].strip()
            
            # Estimate tokens (rough calculation)
            tokens_used = len(content.split())
            
            return GenerationResult(
                content=content,
                metadata={"model": self.config.model_name, "provider": "huggingface"},
                tokens_used=tokens_used,
                model_used=self.config.model_name
            )
            
        except Exception as e:
            raise Exception(f"HuggingFace generation failed: {e}")
    
    async def _generate_replicate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Generate content using Replicate"""
        try:
            # Use a text generation model on Replicate
            model_name = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c8133"
            
            output = await replicate.async_run(
                model_name,
                input={"prompt": prompt, "max_tokens": self.config.max_tokens}
            )
            
            content = "".join(output)
            tokens_used = len(content.split())  # Rough estimation
            
            return GenerationResult(
                content=content,
                metadata={"model": model_name, "provider": "replicate"},
                tokens_used=tokens_used,
                model_used=model_name
            )
            
        except Exception as e:
            raise Exception(f"Replicate generation failed: {e}")


class RefactoredLangChainEmailService:
    """
    Refactored LangChain email service with modern architecture and advanced features.
    """
    
    def __init__(self, config: LangChainConfig):
        """
        Initialize the refactored LangChain email service.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.model_manager = ModelManager(config)
        
        # Advanced components
        self.cache = TTLCache(maxsize=config.cache_size, ttl=config.cache_ttl)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Circuit breakers
        self.content_generation_circuit_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=30)
        self.personalization_circuit_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)
        
        # Statistics
        self.stats = {
            "content_generated": 0,
            "personalizations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_tokens_used": 0
        }
        
        logger.info("Refactored LangChain Email Service initialized", config=config.__dict__)
    
    async def initialize(self) -> None:
        """Initialize the service"""
        try:
            await self.model_manager.initialize()
            logger.info("LangChain service initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize LangChain service", error=str(e))
            raise
    
    @pybreaker.circuit_breaker(fail_max=3, reset_timeout=30)
    async def create_sequence(
        self,
        name: str,
        target_audience: str,
        goals: List[str],
        tone: str = "professional",
        templates: List[EmailTemplate] = None
    ) -> Dict[str, Any]:
        """Create a complete email sequence using AI"""
        start_time = time.time()
        
        try:
            # Generate sequence structure
            prompt_template = self.model_manager.get_prompt_template("sequence_creation")
            prompt = prompt_template.format(
                name=name,
                audience=target_audience,
                goals=", ".join(goals),
                tone=tone,
                steps_count=5  # Default number of steps
            )
            
            result = await self.model_manager.generate_content(prompt)
            
            if not result.success:
                raise Exception(f"Failed to generate sequence: {result.error}")
            
            # Parse the generated sequence
            sequence_data = self._parse_sequence_generation(result.content)
            
            # Add metadata
            sequence_data.update({
                "name": name,
                "target_audience": target_audience,
                "goals": goals,
                "tone": tone,
                "templates": templates or [],
                "generation_time": result.generation_time,
                "tokens_used": result.tokens_used
            })
            
            # Update statistics
            self.stats["content_generated"] += 1
            self.stats["total_tokens_used"] += result.tokens_used
            
            logger.info("Sequence created successfully",
                       sequence_name=name,
                       generation_time=result.generation_time,
                       tokens_used=result.tokens_used)
            
            return sequence_data
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to create sequence", error=str(e))
            raise
    
    @pybreaker.circuit_breaker(fail_max=5, reset_timeout=60)
    async def personalize_content(
        self,
        content: str,
        subscriber: Subscriber,
        variables: Dict[str, Any]
    ) -> str:
        """Personalize content for a specific subscriber"""
        start_time = time.time()
        
        try:
            # Create personalization context
            context = PersonalizationContext(
                subscriber=subscriber,
                sequence=None,  # Will be set if available
                variables=variables
            )
            
            # Generate personalized content
            prompt_template = self.model_manager.get_prompt_template("personalization")
            prompt = prompt_template.format(
                content=content,
                subscriber_data=self._format_subscriber_data(subscriber),
                variables=str(variables)
            )
            
            result = await self.model_manager.generate_content(prompt)
            
            if not result.success:
                raise Exception(f"Failed to personalize content: {result.error}")
            
            # Update statistics
            self.stats["personalizations"] += 1
            self.stats["total_tokens_used"] += result.tokens_used
            
            logger.info("Content personalized successfully",
                       subscriber_id=str(subscriber.id),
                       generation_time=result.generation_time,
                       tokens_used=result.tokens_used)
            
            return result.content
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to personalize content", error=str(e))
            raise
    
    async def generate_email_content(
        self,
        target_audience: str,
        goals: List[str],
        tone: str = "professional",
        context: str = ""
    ) -> str:
        """Generate email content using AI"""
        start_time = time.time()
        
        try:
            prompt_template = self.model_manager.get_prompt_template("email_generation")
            prompt = prompt_template.format(
                audience=target_audience,
                goals=", ".join(goals),
                tone=tone,
                context=context
            )
            
            result = await self.model_manager.generate_content(prompt)
            
            if not result.success:
                raise Exception(f"Failed to generate email content: {result.error}")
            
            # Update statistics
            self.stats["content_generated"] += 1
            self.stats["total_tokens_used"] += result.tokens_used
            
            logger.info("Email content generated successfully",
                       generation_time=result.generation_time,
                       tokens_used=result.tokens_used)
            
            return result.content
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to generate email content", error=str(e))
            raise
    
    async def generate_subject_lines(
        self,
        email_content: str,
        target_audience: str,
        goals: List[str]
    ) -> List[str]:
        """Generate multiple subject lines for A/B testing"""
        start_time = time.time()
        
        try:
            prompt_template = self.model_manager.get_prompt_template("subject_line")
            prompt = prompt_template.format(
                email_content=email_content,
                audience=target_audience,
                goals=", ".join(goals)
            )
            
            result = await self.model_manager.generate_content(prompt)
            
            if not result.success:
                raise Exception(f"Failed to generate subject lines: {result.error}")
            
            # Parse subject lines from the generated content
            subject_lines = self._parse_subject_lines(result.content)
            
            # Update statistics
            self.stats["content_generated"] += 1
            self.stats["total_tokens_used"] += result.tokens_used
            
            logger.info("Subject lines generated successfully",
                       count=len(subject_lines),
                       generation_time=result.generation_time,
                       tokens_used=result.tokens_used)
            
            return subject_lines
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to generate subject lines", error=str(e))
            raise
    
    async def evaluate_condition(
        self,
        condition_expression: str,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> bool:
        """Evaluate a condition expression using AI"""
        try:
            # Create context for condition evaluation
            context = {
                "sequence": sequence.to_dict(),
                "step": step.to_dict() if step else {},
                "condition": condition_expression
            }
            
            prompt = f"""
            Evaluate the following condition for an email sequence:
            
            Condition: {condition_expression}
            Sequence Context: {context}
            
            Please evaluate this condition and return only 'true' or 'false'.
            Consider the sequence context and subscriber data when evaluating.
            """
            
            result = await self.model_manager.generate_content(prompt)
            
            if not result.success:
                raise Exception(f"Failed to evaluate condition: {result.error}")
            
            # Parse the result
            response = result.content.strip().lower()
            return response in ["true", "yes", "1"]
            
        except Exception as e:
            logger.error("Failed to evaluate condition", error=str(e))
            return False
    
    async def execute_action(
        self,
        action_type: str,
        action_data: Dict[str, Any],
        sequence: EmailSequence,
        step: SequenceStep
    ) -> Dict[str, Any]:
        """Execute an action using AI"""
        try:
            context = {
                "sequence": sequence.to_dict(),
                "step": step.to_dict() if step else {},
                "action_type": action_type,
                "action_data": action_data
            }
            
            prompt = f"""
            Execute the following action for an email sequence:
            
            Action Type: {action_type}
            Action Data: {action_data}
            Sequence Context: {context}
            
            Please execute this action and return the result as JSON.
            """
            
            result = await self.model_manager.generate_content(prompt)
            
            if not result.success:
                raise Exception(f"Failed to execute action: {result.error}")
            
            # Try to parse JSON result
            try:
                import json
                return json.loads(result.content)
            except json.JSONDecodeError:
                return {"result": result.content, "success": True}
            
        except Exception as e:
            logger.error("Failed to execute action", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def execute_webhook(
        self,
        webhook_url: str,
        webhook_data: Dict[str, Any],
        sequence: EmailSequence,
        step: SequenceStep
    ) -> Dict[str, Any]:
        """Execute a webhook with AI-enhanced data"""
        try:
            # Enhance webhook data with AI insights
            enhanced_data = await self._enhance_webhook_data(webhook_data, sequence, step)
            
            # Execute webhook (this would be implemented with actual HTTP client)
            # For now, we'll simulate the execution
            result = {
                "url": webhook_url,
                "data": enhanced_data,
                "success": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("Webhook executed successfully", url=webhook_url)
            return result
            
        except Exception as e:
            logger.error("Failed to execute webhook", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _enhance_webhook_data(
        self,
        webhook_data: Dict[str, Any],
        sequence: EmailSequence,
        step: SequenceStep
    ) -> Dict[str, Any]:
        """Enhance webhook data with AI insights"""
        try:
            context = {
                "sequence": sequence.to_dict(),
                "step": step.to_dict() if step else {},
                "original_data": webhook_data
            }
            
            prompt = f"""
            Enhance the following webhook data with AI insights:
            
            Original Data: {webhook_data}
            Context: {context}
            
            Please enhance the data with relevant insights and return as JSON.
            """
            
            result = await self.model_manager.generate_content(prompt)
            
            if not result.success:
                return webhook_data  # Return original data if enhancement fails
            
            # Try to parse enhanced data
            try:
                import json
                enhanced_data = json.loads(result.content)
                return {**webhook_data, **enhanced_data}
            except json.JSONDecodeError:
                return webhook_data
            
        except Exception as e:
            logger.error("Failed to enhance webhook data", error=str(e))
            return webhook_data
    
    def _format_subscriber_data(self, subscriber: Subscriber) -> str:
        """Format subscriber data for AI processing"""
        return f"""
        Name: {subscriber.first_name} {subscriber.last_name}
        Email: {subscriber.email}
        Status: {subscriber.status.value}
        Preferences: {subscriber.preferences}
        Engagement Score: {getattr(subscriber, 'engagement_score', 'N/A')}
        """
    
    def _parse_sequence_generation(self, content: str) -> Dict[str, Any]:
        """Parse generated sequence content"""
        try:
            # This is a simplified parser - in production, you'd want more robust parsing
            lines = content.split('\n')
            steps = []
            
            current_step = None
            for line in lines:
                line = line.strip()
                if line.startswith('Step') or line.startswith('Email'):
                    if current_step:
                        steps.append(current_step)
                    current_step = {
                        "step_type": "email",
                        "order": len(steps) + 1,
                        "subject": "",
                        "content": "",
                        "delay_hours": 24,
                        "is_active": True
                    }
                elif current_step and line:
                    if not current_step["subject"]:
                        current_step["subject"] = line
                    else:
                        current_step["content"] += line + "\n"
            
            if current_step:
                steps.append(current_step)
            
            return {
                "steps": steps,
                "personalization_variables": {
                    "name": "{first_name}",
                    "email": "{email}",
                    "company": "{company}"
                }
            }
            
        except Exception as e:
            logger.error("Failed to parse sequence generation", error=str(e))
            return {
                "steps": [],
                "personalization_variables": {}
            }
    
    def _parse_subject_lines(self, content: str) -> List[str]:
        """Parse subject lines from generated content"""
        try:
            lines = content.split('\n')
            subject_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and len(line) < 100:
                    # Clean up the line
                    line = line.replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '')
                    line = line.strip()
                    if line:
                        subject_lines.append(line)
            
            return subject_lines[:5]  # Return up to 5 subject lines
            
        except Exception as e:
            logger.error("Failed to parse subject lines", error=str(e))
            return []
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        return {
            "content_generated": self.stats["content_generated"],
            "personalizations": self.stats["personalizations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "errors": self.stats["errors"],
            "total_tokens_used": self.stats["total_tokens_used"],
            "cache_stats": {
                "size": len(self.cache),
                "hits": self.stats["cache_hits"],
                "misses": self.stats["cache_misses"]
            },
            "model_stats": {
                "provider": self.config.provider.value,
                "model_name": self.config.model_name,
                "circuit_breaker_status": {
                    "content_generation": self.content_generation_circuit_breaker.current_state,
                    "personalization": self.personalization_circuit_breaker.current_state
                }
            }
        } 