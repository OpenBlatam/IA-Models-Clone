from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from openai import AsyncOpenAI
import uvloop
import orjson
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil
from functools import lru_cache
import pickle
import gzip
from domain.entities import CopywritingRequest, CopywritingResponse
from domain.interfaces import AIService
from typing import Any, List, Dict, Optional
"""
Ultra-Optimized AI Service
==========================

Advanced AI service with GPU acceleration, intelligent caching, and autonomous optimization.
"""


# AI/ML Libraries

# Performance Libraries

# Caching and Optimization


logger = logging.getLogger(__name__)


class DevinAIService(AIService):
    """Ultra-optimized AI service with advanced features."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: str = "/tmp/models",
        max_length: int = 512,
        temperature: float = 0.7,
        use_gpu: bool = True,
        batch_size: int = 4,
        max_workers: int = 8,
        enable_quantization: bool = True,
        enable_compilation: bool = True
    ):
        
    """__init__ function."""
self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.temperature = temperature
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_quantization = enable_quantization
        self.enable_compilation = enable_compilation
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.embedding_model = None
        self.openai_client = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "gpu_usage": 0.0,
            "memory_usage": 0.0,
            "average_response_time": 0.0
        }
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Model cache
        self._model_cache = {}
        self._response_cache = {}
        
        # Optimization flags
        self._initialized = False
        self._optimization_enabled = True
        
        logger.info(f"DevinAIService initialized with model: {model_name}")
    
    async def initialize(self) -> Any:
        """Initialize AI service with optimizations."""
        if self._initialized:
            return
        
        try:
            logger.info("🚀 Initializing ultra-optimized AI service...")
            
            # Set event loop policy for better performance
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            
            # Initialize models
            await self._initialize_models()
            
            # Initialize OpenAI client if API key available
            await self._initialize_openai()
            
            # Run optimization tasks
            await self._run_optimizations()
            
            self._initialized = True
            logger.info("✅ AI service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI service: {e}")
            raise
    
    async def _initialize_models(self) -> Any:
        """Initialize AI models with optimizations."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_fast=True
            )
            
            # Load model with optimizations
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16 if self.use_gpu else torch.float32,
                "device_map": "auto" if self.use_gpu else None
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Apply optimizations
            if self.use_gpu:
                self.model = self.model.cuda()
                
                if self.enable_quantization:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                
                if self.enable_compilation and hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode="reduce-overhead")
            
            # Initialize generator
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32
            )
            
            # Load embedding model for similarity
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("✅ Models loaded with optimizations")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    async def _initialize_openai(self) -> Any:
        """Initialize OpenAI client if API key is available."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.warning(f"⚠️ OpenAI client not available: {e}")
    
    async def _run_optimizations(self) -> Any:
        """Run performance optimizations."""
        try:
            # Warm up models
            await self._warmup_models()
            
            # Preload common prompts
            await self._preload_common_prompts()
            
            # Optimize memory usage
            await self._optimize_memory()
            
            logger.info("✅ Performance optimizations completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Some optimizations failed: {e}")
    
    async def _warmup_models(self) -> Any:
        """Warm up models for better performance."""
        try:
            warmup_text = "Generate a short product description"
            await self.generate_copywriting(
                CopywritingRequest(
                    prompt=warmup_text,
                    style="professional",
                    tone="neutral",
                    length=50
                )
            )
            logger.info("✅ Model warmup completed")
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")
    
    async def _preload_common_prompts(self) -> Any:
        """Preload common prompts for faster response."""
        common_prompts = [
            "product description",
            "marketing copy",
            "social media post",
            "email subject line",
            "landing page headline"
        ]
        
        for prompt in common_prompts:
            cache_key = self._generate_cache_key(prompt)
            if cache_key not in self._response_cache:
                # Pre-generate and cache
                pass
    
    async def _optimize_memory(self) -> Any:
        """Optimize memory usage."""
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        # Set memory limits
        if hasattr(torch, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(0.8)
    
    async def generate_copywriting(self, request: CopywritingRequest) -> CopywritingResponse:
        """Generate copywriting with ultra-optimized performance."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = self._response_cache.get(cache_key)
            
            if cached_response:
                self.performance_metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for request: {request.id}")
                return cached_response
            
            # Generate content
            if self.openai_client and self._should_use_openai(request):
                generated_text = await self._generate_with_openai(request)
            else:
                generated_text = await self._generate_with_local_model(request)
            
            # Post-process
            processed_text = await self._post_process_text(generated_text, request)
            
            # Create response
            processing_time = time.time() - start_time
            response = CopywritingResponse(
                request_id=request.id,
                generated_text=processed_text,
                processing_time=processing_time,
                model_used=self.model_name,
                confidence_score=self._calculate_confidence(processed_text, request)
            )
            
            # Cache response
            self._response_cache[cache_key] = response
            
            # Update metrics
            self._update_metrics(processing_time)
            
            logger.info(f"Generated copywriting in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating copywriting: {e}")
            raise
    
    async def _generate_with_local_model(self, request: CopywritingRequest) -> str:
        """Generate text using local model."""
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(request)
            
            # Generate with optimizations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_sync,
                prompt,
                request.length,
                request.creativity
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error with local model generation: {e}")
            raise
    
    def _generate_sync(self, prompt: str, length: int, creativity: float) -> str:
        """Synchronous generation for thread pool."""
        try:
            # Adjust temperature based on creativity
            temperature = 0.5 + (creativity * 0.5)
            
            # Generate with optimized parameters
            result = self.generator(
                prompt,
                max_length=length + len(prompt.split()),
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract only the new content
            if prompt in generated_text:
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in sync generation: {e}")
            raise
    
    async def _generate_with_openai(self, request: CopywritingRequest) -> str:
        """Generate text using OpenAI API."""
        try:
            prompt = self._prepare_prompt(request)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_system_prompt(request)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.length * 2,  # Approximate tokens
                temperature=request.creativity,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error with OpenAI generation: {e}")
            # Fallback to local model
            return await self._generate_with_local_model(request)
    
    def _prepare_prompt(self, request: CopywritingRequest) -> str:
        """Prepare optimized prompt for generation."""
        style_instructions = {
            "professional": "Write in a professional, business-like tone",
            "casual": "Write in a casual, friendly tone",
            "creative": "Write in a creative, imaginative style",
            "technical": "Write in a technical, detailed manner",
            "persuasive": "Write in a persuasive, compelling style"
        }
        
        tone_instructions = {
            "neutral": "Maintain a neutral tone",
            "enthusiastic": "Use an enthusiastic, excited tone",
            "authoritative": "Use an authoritative, confident tone",
            "friendly": "Use a friendly, approachable tone"
        }
        
        prompt_parts = [
            f"Style: {style_instructions.get(request.style, 'professional')}",
            f"Tone: {tone_instructions.get(request.tone, 'neutral')}",
            f"Length: {request.length} words",
            f"Language: {request.language}"
        ]
        
        if request.target_audience:
            prompt_parts.append(f"Target audience: {request.target_audience}")
        
        if request.keywords:
            prompt_parts.append(f"Keywords to include: {', '.join(request.keywords)}")
        
        prompt_parts.append(f"Task: {request.prompt}")
        
        return "\n".join(prompt_parts)
    
    def _get_system_prompt(self, request: CopywritingRequest) -> str:
        """Get system prompt for OpenAI."""
        return f"""You are an expert copywriter specializing in {request.style} content with a {request.tone} tone. 
        Generate high-quality, engaging copy that meets the specified requirements."""
    
    async def _post_process_text(self, text: str, request: CopywritingRequest) -> str:
        """Post-process generated text for quality improvement."""
        try:
            # Clean up text
            text = text.strip()
            
            # Ensure proper length
            words = text.split()
            if len(words) > request.length:
                text = " ".join(words[:request.length])
            
            # Add keywords if missing
            if request.keywords:
                text = await self._ensure_keywords(text, request.keywords)
            
            # Improve readability
            text = await self._improve_readability(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return text
    
    async def _ensure_keywords(self, text: str, keywords: List[str]) -> str:
        """Ensure keywords are included in text."""
        text_lower = text.lower()
        missing_keywords = [kw for kw in keywords if kw.lower() not in text_lower]
        
        if missing_keywords:
            # Add missing keywords naturally
            for keyword in missing_keywords:
                if len(text.split()) < 100:  # Only for shorter texts
                    text += f" {keyword}"
        
        return text
    
    async def _improve_readability(self, text: str) -> str:
        """Improve text readability."""
        # Basic improvements
        text = text.replace("  ", " ")  # Remove double spaces
        text = text.replace(" .", ".")  # Fix spacing around periods
        text = text.replace(" ,", ",")  # Fix spacing around commas
        
        return text
    
    def _calculate_confidence(self, text: str, request: CopywritingRequest) -> float:
        """Calculate confidence score for generated text."""
        try:
            # Basic confidence calculation
            confidence = 0.8  # Base confidence
            
            # Adjust based on length match
            target_length = request.length
            actual_length = len(text.split())
            length_ratio = min(actual_length / target_length, target_length / actual_length)
            confidence *= length_ratio
            
            # Adjust based on keyword inclusion
            if request.keywords:
                included_keywords = sum(1 for kw in request.keywords if kw.lower() in text.lower())
                keyword_ratio = included_keywords / len(request.keywords)
                confidence *= (0.7 + 0.3 * keyword_ratio)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.7
    
    def _generate_cache_key(self, request: CopywritingRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            "prompt": request.prompt,
            "style": request.style,
            "tone": request.tone,
            "length": request.length,
            "creativity": request.creativity,
            "language": request.language,
            "keywords": sorted(request.keywords) if request.keywords else []
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _should_use_openai(self, request: CopywritingRequest) -> bool:
        """Determine if OpenAI should be used for this request."""
        # Use OpenAI for complex requests or when local model is busy
        return (
            self.openai_client is not None and
            (len(request.prompt) > 200 or request.creativity > 0.8)
        )
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics."""
        self.performance_metrics["total_requests"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        total_requests = self.performance_metrics["total_requests"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # Update resource usage
        if self.use_gpu:
            self.performance_metrics["gpu_usage"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        self.performance_metrics["memory_usage"] = psutil.virtual_memory().percent
    
    async def improve_copywriting(self, text: str, suggestions: List[str]) -> str:
        """Improve existing copywriting text."""
        try:
            # Use OpenAI for improvement if available
            if self.openai_client:
                return await self._improve_with_openai(text, suggestions)
            else:
                return await self._improve_with_local_model(text, suggestions)
                
        except Exception as e:
            logger.error(f"Error improving copywriting: {e}")
            return text
    
    async def _improve_with_openai(self, text: str, suggestions: List[str]) -> str:
        """Improve text using OpenAI."""
        try:
            prompt = f"""Improve the following text based on these suggestions: {', '.join(suggestions)}

Original text: {text}

Improved text:"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert copywriter. Improve the given text based on the suggestions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=len(text.split()) * 2,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI improvement failed: {e}")
            return text
    
    async def _improve_with_local_model(self, text: str, suggestions: List[str]) -> str:
        """Improve text using local model."""
        try:
            prompt = f"Improve this text: {text}\nSuggestions: {', '.join(suggestions)}\nImproved version:"
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_sync,
                prompt,
                len(text.split()) + 50,
                0.3
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Local model improvement failed: {e}")
            return text
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for various metrics."""
        try:
            analysis = {
                "word_count": len(text.split()),
                "character_count": len(text),
                "sentence_count": text.count('.') + text.count('!') + text.count('?'),
                "average_word_length": np.mean([len(word) for word in text.split()]),
                "readability_score": self._calculate_readability(text),
                "sentiment_score": await self._analyze_sentiment(text),
                "keyword_density": self._calculate_keyword_density(text),
                "complexity_score": self._calculate_complexity(text)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {}
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score."""
        try:
            sentences = text.split('.')
            words = text.split()
            syllables = sum(self._count_syllables(word) for word in words)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Flesch Reading Ease
            flesch_score = 206.835 - (1.015 * len(words) / len(sentences)) - (84.6 * syllables / len(words))
            return max(0.0, min(100.0, flesch_score))
            
        except Exception:
            return 50.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
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
        if count == 0:
            count = 1
        return count
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment."""
        try:
            # Simple sentiment analysis
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total = positive_count + negative_count
            if total == 0:
                return 0.0
            
            return (positive_count - negative_count) / total
            
        except Exception:
            return 0.0
    
    def _calculate_keyword_density(self, text: str) -> Dict[str, float]:
        """Calculate keyword density."""
        try:
            words = text.lower().split()
            word_count = len(words)
            
            if word_count == 0:
                return {}
            
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            return {word: freq / word_count for word, freq in word_freq.items() if freq > 1}
            
        except Exception:
            return {}
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        try:
            words = text.split()
            if not words:
                return 0.0
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words])
            
            # Unique word ratio
            unique_ratio = len(set(words)) / len(words)
            
            # Sentence length
            sentences = text.split('.')
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            
            # Complexity score (0-1, higher = more complex)
            complexity = (avg_word_length / 10 + unique_ratio + avg_sentence_length / 20) / 3
            return min(1.0, complexity)
            
        except Exception:
            return 0.5
    
    async def get_suggestions(self, text: str, context: Dict[str, Any]) -> List[str]:
        """Get improvement suggestions for text."""
        try:
            suggestions = []
            
            # Analyze text
            analysis = await self.analyze_text(text)
            
            # Generate suggestions based on analysis
            if analysis.get("readability_score", 100) < 60:
                suggestions.append("Consider using shorter sentences and simpler words for better readability")
            
            if analysis.get("sentiment_score", 0) < -0.3:
                suggestions.append("Consider adding more positive language to improve sentiment")
            
            if analysis.get("complexity_score", 0.5) > 0.7:
                suggestions.append("Consider simplifying the language for broader audience appeal")
            
            if len(text.split()) < 50:
                suggestions.append("Consider adding more details and examples to make the content more comprehensive")
            
            # Add context-specific suggestions
            if context.get("target_audience"):
                suggestions.append(f"Tailor the language to better match your {context['target_audience']} audience")
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return ["Consider reviewing the content for clarity and impact"]
    
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate and analyze prompt."""
        try:
            validation = {
                "is_valid": True,
                "length": len(prompt),
                "word_count": len(prompt.split()),
                "has_clear_instructions": self._check_clear_instructions(prompt),
                "has_target_audience": "audience" in prompt.lower() or "target" in prompt.lower(),
                "has_style_guidance": any(word in prompt.lower() for word in ["style", "tone", "voice"]),
                "suggestions": []
            }
            
            # Generate suggestions
            if validation["length"] < 10:
                validation["suggestions"].append("Prompt is too short. Add more details for better results.")
                validation["is_valid"] = False
            
            if not validation["has_clear_instructions"]:
                validation["suggestions"].append("Add clear instructions about what you want to achieve.")
            
            if not validation["has_target_audience"]:
                validation["suggestions"].append("Specify your target audience for more targeted content.")
            
            if not validation["has_style_guidance"]:
                validation["suggestions"].append("Include style, tone, or voice preferences for better results.")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating prompt: {e}")
            return {"is_valid": False, "error": str(e)}
    
    def _check_clear_instructions(self, prompt: str) -> bool:
        """Check if prompt has clear instructions."""
        instruction_words = ["create", "write", "generate", "describe", "explain", "tell", "make"]
        return any(word in prompt.lower() for word in instruction_words)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model."""
        try:
            info = {
                "model_name": self.model_name,
                "model_type": "transformer",
                "max_length": self.max_length,
                "temperature": self.temperature,
                "use_gpu": self.use_gpu,
                "batch_size": self.batch_size,
                "enable_quantization": self.enable_quantization,
                "enable_compilation": self.enable_compilation,
                "performance_metrics": self.performance_metrics,
                "cache_size": len(self._response_cache),
                "initialized": self._initialized
            }
            
            # Add GPU info if available
            if self.use_gpu:
                info["gpu_info"] = {
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved()
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    async def is_available(self) -> bool:
        """Check if AI service is available."""
        return self._initialized and self.model is not None
    
    async def cleanup(self) -> Any:
        """Cleanup AI service resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            self._initialized = False
            logger.info("AI service cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up AI service: {e}") 