"""
LLM Best Practices Implementation for AI History Comparison System
Mejores Prácticas de LLM para el Sistema de Comparación de Historial de IA
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timezone

# LLM Libraries
import openai
from anthropic import Anthropic
import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Performance and Caching
from functools import lru_cache
import redis
import pickle

# =============================================================================
# 1. CONFIGURACIÓN Y GESTIÓN DE MODELOS
# =============================================================================

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class ModelType(Enum):
    """Model types for different tasks"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour

class LLMManager:
    """Centralized LLM management with best practices"""
    
    def __init__(self):
        self.configs: Dict[str, LLMConfig] = {}
        self.clients: Dict[str, Any] = {}
        self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
    def register_model(self, name: str, config: LLMConfig):
        """Register a new LLM model configuration"""
        self.configs[name] = config
        self._initialize_client(name, config)
        
    def _initialize_client(self, name: str, config: LLMConfig):
        """Initialize client for the given configuration"""
        try:
            if config.provider == LLMProvider.OPENAI:
                self.clients[name] = openai.AsyncOpenAI(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    timeout=config.timeout
                )
            elif config.provider == LLMProvider.ANTHROPIC:
                self.clients[name] = Anthropic(
                    api_key=config.api_key,
                    timeout=config.timeout
                )
            elif config.provider == LLMProvider.GOOGLE:
                genai.configure(api_key=config.api_key)
                self.clients[name] = genai.GenerativeModel(config.model_name)
            elif config.provider == LLMProvider.HUGGINGFACE:
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.clients[name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "pipeline": pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if torch.cuda.is_available() else -1
                    )
                }
                
            self.logger.info(f"✅ Initialized {config.provider.value} model: {name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model {name}: {str(e)}")
            raise

# =============================================================================
# 2. PROMPT ENGINEERING Y TEMPLATES
# =============================================================================

class PromptTemplate:
    """Reusable prompt templates with best practices"""
    
    @staticmethod
    def content_analysis_prompt(content: str, analysis_type: str = "comprehensive") -> str:
        """Generate content analysis prompt"""
        base_prompt = f"""
You are an expert content analyst. Analyze the following content and provide a comprehensive assessment.

Content to analyze:
"{content}"

Please provide analysis in the following JSON format:
{{
    "readability_score": 0.0-1.0,
    "sentiment_score": -1.0 to 1.0,
    "complexity_score": 0.0-1.0,
    "topic_diversity": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "key_themes": ["theme1", "theme2"],
    "strengths": ["strength1", "strength2"],
    "improvements": ["improvement1", "improvement2"],
    "overall_quality": 0.0-1.0,
    "confidence": 0.0-1.0
}}

Analysis type: {analysis_type}
"""
        return base_prompt.strip()
    
    @staticmethod
    def comparison_prompt(content1: str, content2: str) -> str:
        """Generate content comparison prompt"""
        return f"""
You are an expert content comparison analyst. Compare the following two pieces of content and provide detailed analysis.

Content 1:
"{content1}"

Content 2:
"{content2}"

Please provide comparison in the following JSON format:
{{
    "similarity_score": 0.0-1.0,
    "quality_difference": {{
        "content1_score": 0.0-1.0,
        "content2_score": 0.0-1.0,
        "difference": -1.0 to 1.0
    }},
    "style_differences": ["difference1", "difference2"],
    "content_differences": ["difference1", "difference2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "trend_direction": "improving|declining|stable",
    "confidence": 0.0-1.0
}}
"""
    
    @staticmethod
    def trend_analysis_prompt(contents: List[str], timeframes: List[str]) -> str:
        """Generate trend analysis prompt"""
        content_list = "\n".join([f"Timeframe {i+1} ({timeframes[i]}): {content}" 
                                 for i, content in enumerate(contents)])
        
        return f"""
You are an expert trend analyst. Analyze the following content over time and identify patterns and trends.

Content over time:
{content_list}

Please provide trend analysis in the following JSON format:
{{
    "overall_trend": "improving|declining|stable|volatile",
    "trend_strength": 0.0-1.0,
    "key_metrics": {{
        "readability_trend": "improving|declining|stable",
        "sentiment_trend": "improving|declining|stable",
        "complexity_trend": "improving|declining|stable"
    }},
    "significant_changes": ["change1", "change2"],
    "predictions": {{
        "next_period_quality": 0.0-1.0,
        "confidence": 0.0-1.0
    }},
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""

# =============================================================================
# 3. CACHING Y OPTIMIZACIÓN
# =============================================================================

class LLMCache:
    """Intelligent caching for LLM responses"""
    
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)
    
    def _generate_cache_key(self, prompt: str, model_name: str, config: Dict[str, Any]) -> str:
        """Generate cache key from prompt and configuration"""
        # Create hash of prompt + model + config
        content = f"{prompt}:{model_name}:{json.dumps(config, sort_keys=True)}"
        return f"llm_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get(self, prompt: str, model_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        if not config.get('cache_enabled', True):
            return None
            
        try:
            cache_key = self._generate_cache_key(prompt, model_name, config)
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                self.logger.info(f"🎯 Cache hit for model {model_name}")
                return json.loads(cached_data)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {str(e)}")
            
        return None
    
    async def set(self, prompt: str, model_name: str, config: Dict[str, Any], 
                  response: Dict[str, Any], ttl: Optional[int] = None):
        """Cache response"""
        if not config.get('cache_enabled', True):
            return
            
        try:
            cache_key = self._generate_cache_key(prompt, model_name, config)
            ttl = ttl or config.get('cache_ttl', self.default_ttl)
            
            self.redis.setex(
                cache_key,
                ttl,
                json.dumps(response, default=str)
            )
            self.logger.info(f"💾 Cached response for model {model_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache set error: {str(e)}")

# =============================================================================
# 4. MANEJO DE ERRORES Y RETRY
# =============================================================================

class LLMError(Exception):
    """Custom LLM error class"""
    def __init__(self, message: str, error_code: str = None, retry_after: int = None):
        self.message = message
        self.error_code = error_code
        self.retry_after = retry_after
        super().__init__(self.message)

class RetryManager:
    """Intelligent retry management for LLM calls"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.logger.error(f"❌ Max retries exceeded for {func.__name__}")
                    break
                
                # Calculate delay with exponential backoff
                delay = self.base_delay * (self.backoff_factor ** attempt)
                self.logger.warning(f"⚠️ Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                
                await asyncio.sleep(delay)
        
        raise LLMError(
            f"Function {func.__name__} failed after {self.max_retries} retries",
            error_code="MAX_RETRIES_EXCEEDED"
        )

# =============================================================================
# 5. PROCESAMIENTO ASÍNCRONO Y BATCHING
# =============================================================================

class AsyncLLMProcessor:
    """Asynchronous LLM processing with batching"""
    
    def __init__(self, llm_manager: LLMManager, cache: LLMCache, retry_manager: RetryManager):
        self.llm_manager = llm_manager
        self.cache = cache
        self.retry_manager = retry_manager
        self.logger = logging.getLogger(__name__)
    
    async def process_single(self, prompt: str, model_name: str, 
                           config: LLMConfig) -> Dict[str, Any]:
        """Process single prompt with caching and retry"""
        
        # Check cache first
        cached_response = await self.cache.get(prompt, model_name, asdict(config))
        if cached_response:
            return cached_response
        
        # Process with retry
        async def _call_llm():
            return await self._call_llm_provider(prompt, model_name, config)
        
        response = await self.retry_manager.execute_with_retry(_call_llm)
        
        # Cache response
        await self.cache.set(prompt, model_name, asdict(config), response)
        
        return response
    
    async def process_batch(self, prompts: List[str], model_name: str, 
                          config: LLMConfig, max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple prompts concurrently with rate limiting"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_with_semaphore(prompt: str):
            async with semaphore:
                return await self.process_single(prompt, model_name, config)
        
        tasks = [_process_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Error processing prompt {i}: {str(result)}")
                processed_results.append({
                    "error": str(result),
                    "success": False
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _call_llm_provider(self, prompt: str, model_name: str, config: LLMConfig) -> Dict[str, Any]:
        """Call the appropriate LLM provider"""
        client = self.llm_manager.clients[model_name]
        provider = config.provider
        
        start_time = time.time()
        
        try:
            if provider == LLMProvider.OPENAI:
                response = await client.chat.completions.create(
                    model=config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty
                )
                content = response.choices[0].message.content
                
            elif provider == LLMProvider.ANTHROPIC:
                response = await client.messages.create(
                    model=config.model_name,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                
            elif provider == LLMProvider.GOOGLE:
                response = await client.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p
                    )
                )
                content = response.text
                
            elif provider == LLMProvider.HUGGINGFACE:
                # Synchronous call for HuggingFace
                pipeline = client["pipeline"]
                response = pipeline(
                    prompt,
                    max_length=config.max_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=client["tokenizer"].eos_token_id
                )
                content = response[0]["generated_text"]
            
            # Parse JSON response
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, wrap in a structure
                parsed_content = {
                    "raw_response": content,
                    "parsed": False
                }
            
            processing_time = time.time() - start_time
            
            return {
                "content": parsed_content,
                "model": model_name,
                "provider": provider.value,
                "processing_time": processing_time,
                "tokens_used": getattr(response, 'usage', {}).get('total_tokens', 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ LLM call failed: {str(e)}")
            
            return {
                "error": str(e),
                "model": model_name,
                "provider": provider.value,
                "processing_time": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": False
            }

# =============================================================================
# 6. MONITOREO Y MÉTRICAS
# =============================================================================

class LLMMonitor:
    """Monitor LLM usage and performance"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def log_usage(self, model_name: str, provider: str, 
                       tokens_used: int, processing_time: float, 
                       success: bool, error: str = None):
        """Log LLM usage metrics"""
        try:
            timestamp = datetime.now(timezone.utc)
            date_key = timestamp.strftime("%Y-%m-%d")
            
            # Log daily usage
            usage_key = f"llm_usage:{date_key}:{model_name}"
            self.redis.hincrby(usage_key, "total_calls", 1)
            self.redis.hincrby(usage_key, "total_tokens", tokens_used)
            self.redis.hincrbyfloat(usage_key, "total_time", processing_time)
            
            if success:
                self.redis.hincrby(usage_key, "successful_calls", 1)
            else:
                self.redis.hincrby(usage_key, "failed_calls", 1)
                if error:
                    self.redis.lpush(f"llm_errors:{date_key}", json.dumps({
                        "model": model_name,
                        "provider": provider,
                        "error": error,
                        "timestamp": timestamp.isoformat()
                    }))
            
            # Set expiration (keep for 30 days)
            self.redis.expire(usage_key, 30 * 24 * 3600)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log usage metrics: {str(e)}")
    
    async def get_usage_stats(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for a model"""
        try:
            stats = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_tokens": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "success_rate": 0.0
            }
            
            for i in range(days):
                date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
                usage_key = f"llm_usage:{date}:{model_name}"
                
                if self.redis.exists(usage_key):
                    data = self.redis.hgetall(usage_key)
                    stats["total_calls"] += int(data.get("total_calls", 0))
                    stats["successful_calls"] += int(data.get("successful_calls", 0))
                    stats["failed_calls"] += int(data.get("failed_calls", 0))
                    stats["total_tokens"] += int(data.get("total_tokens", 0))
                    stats["total_time"] += float(data.get("total_time", 0))
            
            if stats["total_calls"] > 0:
                stats["average_time"] = stats["total_time"] / stats["total_calls"]
                stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get usage stats: {str(e)}")
            return {}

# =============================================================================
# 7. FACTORY Y CONFIGURACIÓN
# =============================================================================

class LLMFactory:
    """Factory for creating LLM instances with best practices"""
    
    @staticmethod
    def create_llm_manager() -> LLMManager:
        """Create LLM manager with default configurations"""
        manager = LLMManager()
        
        # Register default models (configure with your API keys)
        default_configs = [
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                api_key="your-openai-key",
                max_tokens=4000,
                temperature=0.7
            ),
            LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                api_key="your-anthropic-key",
                max_tokens=4000,
                temperature=0.7
            ),
            LLMConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-pro",
                api_key="your-google-key",
                max_tokens=4000,
                temperature=0.7
            )
        ]
        
        for i, config in enumerate(default_configs):
            manager.register_model(f"model_{i}", config)
        
        return manager
    
    @staticmethod
    def create_processor() -> AsyncLLMProcessor:
        """Create async processor with all components"""
        manager = LLMFactory.create_llm_manager()
        cache = LLMCache(redis.Redis(host='localhost', port=6379, db=0))
        retry_manager = RetryManager()
        
        return AsyncLLMProcessor(manager, cache, retry_manager)

# =============================================================================
# 8. EJEMPLO DE USO
# =============================================================================

async def example_usage():
    """Example of using the LLM best practices"""
    
    # Create processor
    processor = LLMFactory.create_processor()
    
    # Get model configuration
    model_name = "model_0"  # OpenAI GPT-4
    config = processor.llm_manager.configs[model_name]
    
    # Example 1: Single content analysis
    content = "This is a sample content for analysis."
    prompt = PromptTemplate.content_analysis_prompt(content)
    
    result = await processor.process_single(prompt, model_name, config)
    print("Single analysis result:", result)
    
    # Example 2: Batch processing
    contents = [
        "First content piece",
        "Second content piece",
        "Third content piece"
    ]
    
    prompts = [PromptTemplate.content_analysis_prompt(content) for content in contents]
    results = await processor.process_batch(prompts, model_name, config, max_concurrent=3)
    print("Batch results:", results)
    
    # Example 3: Content comparison
    content1 = "Original content"
    content2 = "Modified content"
    comparison_prompt = PromptTemplate.comparison_prompt(content1, content2)
    
    comparison_result = await processor.process_single(comparison_prompt, model_name, config)
    print("Comparison result:", comparison_result)

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())







