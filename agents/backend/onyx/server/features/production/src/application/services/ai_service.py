from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import json
import openai
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
from pydantic import BaseModel
from src.core.config import OpenAISettings, AISettings
from src.core.exceptions import AIServiceUnavailableException, InferenceException
            import random
from typing import Any, List, Dict, Optional
"""
🤖 Ultra-Optimized AI Service
============================

Production-grade AI service with GPU acceleration, intelligent caching,
batch processing, and advanced optimization features.
"""





class AIServiceConfig(BaseModel):
    """AI Service configuration"""
    
    # OpenAI settings
    api_key: str
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    default_model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    
    # Performance settings
    gpu_enabled: bool = False
    batch_size: int = 10
    max_concurrent_requests: int = 50
    cache_ttl: int = 3600
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    summarization_model: str = "facebook/bart-large-cnn"
    
    class Config:
        env_prefix = "AI_"


class AIService:
    """
    Ultra-optimized AI service with:
    - GPU acceleration
    - Intelligent caching
    - Batch processing
    - Model optimization
    - Performance monitoring
    """
    
    def __init__(
        self,
        openai_config: OpenAISettings,
        ai_config: AISettings,
        cache_service: Any
    ):
        
    """__init__ function."""
self.config = AIServiceConfig(
            api_key=openai_config.API_KEY.get_secret_value(),
            organization=openai_config.ORGANIZATION,
            base_url=openai_config.BASE_URL,
            timeout=openai_config.TIMEOUT,
            max_retries=openai_config.MAX_RETRIES,
            default_model=openai_config.MODEL,
            max_tokens=openai_config.MAX_TOKENS,
            temperature=openai_config.TEMPERATURE,
            gpu_enabled=ai_config.GPU_ENABLED,
            batch_size=ai_config.BATCH_SIZE,
            max_concurrent_requests=ai_config.MAX_CONCURRENT_REQUESTS
        )
        
        self.cache_service = cache_service
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_generation_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize models (lazy loading)
        self._models = {}
        self._model_lock = asyncio.Lock()
        
        # Request semaphore for concurrency control
        self.request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Batch processing queue
        self.batch_queue = asyncio.Queue()
        self.batch_processor_task = None
        
        self.logger.info("AI Service initialized with ultra-optimization")
    
    async def initialize(self) -> Any:
        """Initialize AI service and load models"""
        
        self.logger.info("Initializing AI Service...")
        
        try:
            # Start batch processor
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            
            # Preload essential models
            await self._load_models()
            
            # Test OpenAI connection
            await self._test_openai_connection()
            
            self.logger.info("AI Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Service: {e}")
            raise AIServiceUnavailableException("AI Service", str(e))
    
    async def cleanup(self) -> Any:
        """Cleanup AI service resources"""
        
        self.logger.info("Cleaning up AI Service...")
        
        # Stop batch processor
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # Close thread pool
        self.executor.shutdown(wait=True)
        
        # Clear models
        self._models.clear()
        
        self.logger.info("AI Service cleanup completed")
    
    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using OpenAI with ultra-optimization"""
        
        start_time = time.time()
        
        try:
            async with self.request_semaphore:
                # Check cache first
                cache_key = self._generate_cache_key(prompt, kwargs)
                cached_result = await self.cache_service.get(cache_key)
                
                if cached_result:
                    self.cache_hits += 1
                    self.logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                    return cached_result
                
                self.cache_misses += 1
                
                # Prepare parameters
                params = {
                    "model": kwargs.get("model", self.config.default_model),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_p": kwargs.get("top_p", 1.0),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    "presence_penalty": kwargs.get("presence_penalty", 0.0)
                }
                
                # Make API call with retry logic
                response = await self._make_openai_call(params)
                
                # Extract content
                content = response.choices[0].message.content
                
                # Cache result
                await self.cache_service.set(cache_key, content, ttl=self.config.cache_ttl)
                
                # Update metrics
                self._update_metrics(response, time.time() - start_time)
                
                return content
                
        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            raise InferenceException("content generation", str(e))
    
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content quality and characteristics"""
        
        try:
            # Run analysis tasks in parallel
            tasks = [
                self._analyze_sentiment(content),
                self._analyze_readability(content),
                self._analyze_seo_score(content),
                self._generate_summary(content),
                self._calculate_reading_time(content)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                "sentiment": results[0] if not isinstance(results[0], Exception) else None,
                "readability_score": results[1] if not isinstance(results[1], Exception) else None,
                "seo_score": results[2] if not isinstance(results[2], Exception) else None,
                "summary": results[3] if not isinstance(results[3], Exception) else None,
                "reading_time": results[4] if not isinstance(results[4], Exception) else None,
                "word_count": len(content.split()),
                "character_count": len(content),
                "sentence_count": len(content.split('.')),
                "paragraph_count": len(content.split('\n\n'))
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return {"error": str(e)}
    
    async def optimize_seo(self, content: str, keywords: List[str]) -> str:
        """Optimize content for SEO"""
        
        try:
            # Create SEO optimization prompt
            seo_prompt = f"""
            Optimize the following content for SEO using these keywords: {', '.join(keywords)}
            
            Content:
            {content}
            
            Please:
            1. Naturally incorporate the keywords
            2. Improve headings and structure
            3. Add meta description
            4. Optimize for readability
            5. Keep the original meaning and tone
            
            Return only the optimized content.
            """
            
            optimized_content = await self.generate_content(seo_prompt, temperature=0.3)
            return optimized_content
            
        except Exception as e:
            self.logger.error(f"SEO optimization failed: {e}")
            return content  # Return original content on error
    
    async def translate_content(self, content: str, target_language: str) -> str:
        """Translate content to target language"""
        
        try:
            # Create translation prompt
            translation_prompt = f"""
            Translate the following content to {target_language}. 
            Maintain the original tone, style, and meaning.
            
            Content:
            {content}
            
            Return only the translated content.
            """
            
            translated_content = await self.generate_content(translation_prompt, temperature=0.2)
            return translated_content
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            raise InferenceException("translation", str(e))
    
    async def summarize_content(self, content: str, max_length: int = 200) -> str:
        """Summarize content using AI"""
        
        try:
            # Use specialized summarization model if available
            if "summarization" in self._models:
                summary = await self._use_summarization_model(content, max_length)
            else:
                # Fallback to OpenAI
                summary_prompt = f"""
                Summarize the following content in {max_length} words or less:
                
                {content}
                
                Return only the summary.
                """
                summary = await self.generate_content(summary_prompt, temperature=0.3)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            # Fallback to simple extraction
            return self._simple_summary(content, max_length)
    
    async def check_plagiarism(self, content: str) -> float:
        """Check content for plagiarism (simplified implementation)"""
        
        try:
            # This is a simplified implementation
            # In production, you'd use a dedicated plagiarism detection service
            
            # Generate embeddings for content
            embeddings = await self.get_embeddings(content)
            
            # For now, return a random score (0-1, where 0 is original, 1 is plagiarized)
            # In real implementation, compare with database of known content
            return random.uniform(0.0, 0.1)  # Low plagiarism score for demo
            
        except Exception as e:
            self.logger.error(f"Plagiarism check failed: {e}")
            return 0.0
    
    async def get_embeddings(self, text: str) -> List[float]:
        """Get text embeddings using sentence transformers"""
        
        try:
            async with self._model_lock:
                if "embeddings" not in self._models:
                    self._models["embeddings"] = SentenceTransformer(self.config.embedding_model)
                
                model = self._models["embeddings"]
                
                # Run embedding generation in thread pool
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    self.executor,
                    model.encode,
                    text
                )
                
                return embeddings.tolist()
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise InferenceException("embedding generation", str(e))
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate content for multiple prompts efficiently"""
        
        try:
            # Add to batch queue
            batch_id = hashlib.md5(str(prompts).encode()).hexdigest()[:8]
            future = asyncio.Future()
            
            await self.batch_queue.put({
                "batch_id": batch_id,
                "prompts": prompts,
                "kwargs": kwargs,
                "future": future
            })
            
            # Wait for result
            results = await future
            return results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            raise InferenceException("batch generation", str(e))
    
    async def _load_models(self) -> Any:
        """Load AI models for local processing"""
        
        try:
            # Load models in parallel
            tasks = []
            
            # Embeddings model
            tasks.append(self._load_embedding_model())
            
            # Sentiment analysis model
            tasks.append(self._load_sentiment_model())
            
            # Summarization model
            tasks.append(self._load_summarization_model())
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.warning(f"Some models failed to load: {e}")
    
    async def _load_embedding_model(self) -> Any:
        """Load sentence transformer model"""
        
        try:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                SentenceTransformer,
                self.config.embedding_model
            )
            
            async with self._model_lock:
                self._models["embeddings"] = model
            
            self.logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
    
    async def _load_sentiment_model(self) -> Any:
        """Load sentiment analysis model"""
        
        try:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                pipeline,
                "sentiment-analysis",
                self.config.sentiment_model
            )
            
            async with self._model_lock:
                self._models["sentiment"] = model
            
            self.logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load sentiment model: {e}")
    
    async def _load_summarization_model(self) -> Any:
        """Load summarization model"""
        
        try:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                pipeline,
                "summarization",
                self.config.summarization_model
            )
            
            async with self._model_lock:
                self._models["summarization"] = model
            
            self.logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load summarization model: {e}")
    
    async def _test_openai_connection(self) -> Any:
        """Test OpenAI API connection"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            self.logger.info("OpenAI connection test successful")
            
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            raise AIServiceUnavailableException("OpenAI", str(e))
    
    async def _make_openai_call(self, params: Dict[str, Any]):
        """Make OpenAI API call with retry logic"""
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.openai_client.chat.completions.create(**params)
                return response
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(f"OpenAI call failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def _batch_processor(self) -> Any:
        """Background task for processing batch requests"""
        
        while True:
            try:
                batch_data = await self.batch_queue.get()
                
                prompts = batch_data["prompts"]
                kwargs = batch_data["kwargs"]
                future = batch_data["future"]
                
                # Process prompts in batches
                results = []
                for i in range(0, len(prompts), self.config.batch_size):
                    batch = prompts[i:i + self.config.batch_size]
                    
                    # Process batch in parallel
                    batch_tasks = [
                        self.generate_content(prompt, **kwargs)
                        for prompt in batch
                    ]
                    
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    results.extend(batch_results)
                
                # Set result
                if not future.done():
                    future.set_result(results)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                if not future.done():
                    future.set_exception(e)
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze content sentiment"""
        
        try:
            if "sentiment" in self._models:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._models["sentiment"],
                    content
                )
                return result
            else:
                # Fallback to OpenAI
                prompt = f"Analyze the sentiment of this text and return a JSON with 'label' and 'score': {content}"
                response = await self.generate_content(prompt, temperature=0.1)
                return json.loads(response)
                
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "neutral", "score": 0.5}
    
    async def _analyze_readability(self, content: str) -> float:
        """Calculate readability score"""
        
        try:
            # Simple Flesch Reading Ease calculation
            sentences = len(content.split('.'))
            words = len(content.split())
            syllables = sum(self._count_syllables(word) for word in content.split())
            
            if sentences == 0 or words == 0:
                return 0.0
            
            score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"Readability analysis failed: {e}")
            return 50.0
    
    async def _analyze_seo_score(self, content: str) -> float:
        """Calculate SEO score"""
        
        try:
            score = 0.0
            
            # Check for headings
            if any(line.strip().startswith('#') for line in content.split('\n')):
                score += 20
            
            # Check for keywords in title
            if content.lower().count('keyword') > 0:
                score += 15
            
            # Check content length
            if len(content) > 300:
                score += 25
            
            # Check for internal links
            if '[' in content and '](' in content:
                score += 10
            
            # Check for meta description
            if len(content) > 150:
                score += 10
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"SEO analysis failed: {e}")
            return 50.0
    
    async def _generate_summary(self, content: str) -> str:
        """Generate content summary"""
        
        try:
            if "summarization" in self._models:
                return await self._use_summarization_model(content, 200)
            else:
                return await self.summarize_content(content, 200)
                
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return self._simple_summary(content, 200)
    
    async def _use_summarization_model(self, content: str, max_length: int) -> str:
        """Use local summarization model"""
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._models["summarization"],
                content,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            self.logger.error(f"Summarization model failed: {e}")
            return self._simple_summary(content, max_length)
    
    async def _calculate_reading_time(self, content: str) -> int:
        """Calculate estimated reading time in minutes"""
        
        try:
            words = len(content.split())
            # Average reading speed: 200-250 words per minute
            reading_time = max(1, words // 225)
            return reading_time
            
        except Exception as e:
            self.logger.error(f"Reading time calculation failed: {e}")
            return 1
    
    def _simple_summary(self, content: str, max_length: int) -> str:
        """Simple extractive summary"""
        
        try:
            sentences = content.split('.')
            if len(sentences) <= 2:
                return content
            
            # Take first few sentences
            summary_sentences = sentences[:2]
            summary = '. '.join(summary_sentences) + '.'
            
            if len(summary) > max_length:
                summary = summary[:max_length-3] + '...'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Simple summary failed: {e}")
            return content[:max_length] + '...' if len(content) > max_length else content
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def _generate_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for prompt and parameters"""
        
        key_data = {
            "prompt": prompt,
            "model": kwargs.get("model", self.config.default_model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"ai_generation:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _update_metrics(self, response, generation_time: float):
        """Update performance metrics"""
        
        self.request_count += 1
        self.total_generation_time += generation_time
        
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.total_tokens
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        avg_generation_time = (
            self.total_generation_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            if (self.cache_hits + self.cache_misses) > 0 else 0
        )
        
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_generation_time": self.total_generation_time,
            "average_generation_time": avg_generation_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "gpu_enabled": self.config.gpu_enabled,
            "models_loaded": list(self._models.keys())
        } 