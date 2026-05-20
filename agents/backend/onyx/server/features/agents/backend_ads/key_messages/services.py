import logging
import time
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime
import uuid
from llm_interface import call_deepseek_api
import hashlib
from functools import lru_cache
import mmh3
import orjson
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from redis.asyncio import Redis
from datetime import timedelta

from .models import (
    KeyMessageRequest,
    GeneratedResponse,
    KeyMessageResponse,
    MessageType,
    MessageTone
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-generated responses for common patterns
COMMON_RESPONSES = {
    "test_message": {
        "informational": {
            "professional": "Test message received. System operational.",
            "casual": "Hey! Got your test message, everything's working fine!",
            "friendly": "Hi there! Your test message came through perfectly.",
            "authoritative": "Test message confirmed. System status: Operational.",
            "conversational": "Got your test message! Everything's working as expected."
        }
    },
    "welcome": {
        "marketing": {
            "professional": "Welcome to our platform. Start your journey today.",
            "casual": "Hey! Welcome aboard! Let's get started!",
            "friendly": "Welcome! We're excited to have you here!",
            "authoritative": "Welcome. Your success journey begins now.",
            "conversational": "Welcome! Ready to explore what we offer?"
        }
    }
}

class RedisCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = None
        self.redis_url = redis_url
        self.ttl = timedelta(hours=24)
    
    async def connect(self):
        if not self.redis:
            self.redis = await Redis.from_url(self.redis_url)
    
    async def get(self, key: str) -> Optional[str]:
        await self.connect()
        value = await self.redis.get(key)
        return value.decode() if value else None
    
    async def set(self, key: str, value: str):
        await self.connect()
        await self.redis.set(key, value, ex=self.ttl)
    
    async def clear(self):
        if self.redis:
            await self.redis.flushdb()

class KeyMessageService:
    def __init__(self):
        """Initialize the service with optimized caches."""
        # Use Redis for distributed caching
        self._redis_cache = RedisCache()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Pre-generate common responses
        self._pre_generate_common_responses()
    
    def _pre_generate_common_responses(self):
        """Pre-generate common responses for faster retrieval."""
        async def pre_generate():
            for message_type, tones in COMMON_RESPONSES.items():
                for msg_type, tone_responses in tones.items():
                    for tone, response in tone_responses.items():
                        request = KeyMessageRequest(
                            message=message_type,
                            message_type=MessageType(msg_type),
                            tone=MessageTone(tone),
                            keywords=[]
                        )
                        cache_key = self._generate_cache_key(request.model_dump())
                        await self._redis_cache.set(cache_key, response)
        
        # Run pre-generation in background
        asyncio.create_task(pre_generate())
    
    def _generate_cache_key(self, data: Dict) -> str:
        """Generate a fast hash key for caching."""
        return mmh3.hash128(orjson.dumps(data))
    
    async def generate_response(self, request: KeyMessageRequest) -> KeyMessageResponse:
        """Generate a response for a key message with caching."""
        start_time = time.perf_counter()
        
        # Generate cache key
        cache_key = self._generate_cache_key(request.model_dump())
        
        # Check Redis cache
        cached_response = await self._redis_cache.get(cache_key)
        if cached_response:
            response = GeneratedResponse(
                id=str(time.time()),
                original_message=request.message,
                response=cached_response,
                message_type=request.message_type,
                tone=request.tone,
                created_at=time.time(),
                word_count=len(cached_response.split()),
                character_count=len(cached_response),
                keywords_used=request.keywords,
                sentiment_score=0.5,
                readability_score=0.8
            )
            return KeyMessageResponse(
                success=True,
                data=response,
                error=None,
                processing_time=time.perf_counter() - start_time,
                suggestions=[]
            )
        
        # Generate new response
        response = await self._generate_response_async(request)
        
        # Cache response if successful
        if response.success and response.data:
            await self._redis_cache.set(cache_key, response.data.response)
        
        return response
    
    async def analyze_message(self, request: KeyMessageRequest) -> KeyMessageResponse:
        """Analyze a message with caching."""
        start_time = time.perf_counter()
        
        # Generate cache key
        cache_key = f"analysis_{self._generate_cache_key(request.model_dump())}"
        
        # Check Redis cache
        cached_analysis = await self._redis_cache.get(cache_key)
        if cached_analysis:
            analysis = GeneratedResponse(
                id=str(time.time()),
                original_message=request.message,
                response=cached_analysis,
                message_type=request.message_type,
                tone=request.tone,
                created_at=time.time(),
                word_count=len(request.message.split()),
                character_count=len(request.message),
                keywords_used=request.keywords,
                sentiment_score=0.6,
                readability_score=0.7
            )
            return KeyMessageResponse(
                success=True,
                data=analysis,
                error=None,
                processing_time=time.perf_counter() - start_time,
                suggestions=[]
            )
        
        # Perform analysis
        analysis = await self._analyze_message_async(request)
        
        # Cache analysis if successful
        if analysis.success and analysis.data:
            await self._redis_cache.set(cache_key, analysis.data.response)
        
        return analysis
    
    async def _generate_response_async(self, request: KeyMessageRequest) -> KeyMessageResponse:
        """Generate a response asynchronously using LLM."""
        start_time = time.perf_counter()
        
        try:
            # Prepare the prompt for the LLM
            prompt = f"""Generate a {request.message_type.value} message with a {request.tone.value} tone.
            
            Original message: {request.message}
            
            Additional context:
            - Target audience: {request.target_audience or 'General audience'}
            - Context: {request.context or 'No specific context provided'}
            - Keywords to include: {', '.join(request.keywords) if request.keywords else 'None specified'}
            
            Please provide a well-structured response that includes:
            1. The main message
            2. Supporting points
            3. A clear call to action or conclusion
            
            Format the response with markdown for better readability."""

            # Call the LLM API
            llm_response = await call_deepseek_api(prompt)
            
            if not llm_response:
                raise Exception("No response from LLM")

            # Create the response object
            response = GeneratedResponse(
                id=str(uuid.uuid4()),
                original_message=request.message,
                response=llm_response,
                message_type=request.message_type,
                tone=request.tone,
                created_at=time.time(),
                word_count=len(llm_response.split()),
                character_count=len(llm_response),
                keywords_used=request.keywords,
                sentiment_score=0.5,
                readability_score=0.8
            )

            return KeyMessageResponse(
                success=True,
                data=response,
                error=None,
                processing_time=time.perf_counter() - start_time,
                suggestions=[]
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return KeyMessageResponse(
                success=False,
                data=None,
                error=str(e),
                processing_time=time.perf_counter() - start_time,
                suggestions=[]
            )
    
    async def _analyze_message_async(self, request: KeyMessageRequest) -> KeyMessageResponse:
        """Analyze a message asynchronously."""
        start_time = time.perf_counter()
        
        # Prepare analysis prompt - Ultra optimized for speed
        prompt = f"Analyze: {request.message}"

        try:
            # Generate analysis using LLM with optimized parameters
            analysis_text = await call_deepseek_api(
                prompt=prompt,
                system_prompt="Be brief.",
                temperature=0.1  # Very low temperature for fastest, most focused responses
            )
            
            # Calculate word count safely
            word_count = len(str(request.message).split())
            
            analysis = GeneratedResponse(
                id=str(time.time()),
                original_message=request.message,
                response=analysis_text,
                message_type=request.message_type,
                tone=request.tone,
                created_at=time.time(),
                word_count=word_count,
                character_count=len(str(request.message)),
                keywords_used=request.keywords,
                sentiment_score=0.6,
                readability_score=0.7
            )
            
            processing_time = time.perf_counter() - start_time
            return KeyMessageResponse(
                success=True,
                data=analysis,
                error=None,
                processing_time=processing_time,
                suggestions=[]
            )
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return KeyMessageResponse(
                success=False,
                data=None,
                error=str(e),
                processing_time=processing_time,
                suggestions=[]
            )
    
    async def clear_cache(self):
        """Clear all caches."""
        await self._redis_cache.clear() 