"""
LLM Service Core
Core business logic for LLM service using modular architecture.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer
import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from shared.ml import (
    ModelManager,
    InferenceEngine,
    EventBus,
    EventType,
    validate_generation_params,
    validate_model_input,
    ModelService,
    InferenceService,
    ServiceRegistry,
    ModelRepository,
    RepositoryManager,
    LoggingEventListener,
    error_handler,
    timing_decorator,
    # New modules
    CacheManager,
    InputSanitizer,
    RateLimiter,
    AsyncExecutor,
    TokenStreamer,
    HealthChecker,
    HealthStatus,
    get_collector,
    ModelInputValidator,
)


class LLMServiceCore:
    """
    Core LLM service using modular architecture.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.model_manager = ModelManager(
            cache_size=self.config.get("cache_size", 5),
            device=self.device,
            use_fp16=self.config.get("use_fp16", True),
        )
        
        # Initialize repositories
        self.repo_manager = RepositoryManager()
        self.repo_manager.register(
            "models",
            ModelRepository(storage_path=self.config.get("model_storage", "./models"))
        )
        
        # Initialize services
        self.service_registry = ServiceRegistry()
        self.service_registry.register(
            "model",
            ModelService(self.model_manager, config=self.config)
        )
        
        # Initialize event bus
        self.event_bus = EventBus()
        self.event_bus.subscribe_all(LoggingEventListener())
        
        # Inference engines cache
        self._inference_engines: Dict[str, InferenceEngine] = {}
        
        # New modules: Cache, Security, Async, Health, Metrics
        self.cache = CacheManager(
            cache_type="lru",
            max_size=self.config.get("cache_max_size", 1000),
            default_ttl=self.config.get("cache_ttl", 3600),
        )
        self.sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get("rate_limit_max", 100),
            window_seconds=self.config.get("rate_limit_window", 60),
        )
        self.async_executor = AsyncExecutor(
            max_workers=self.config.get("async_workers", 4),
        )
        self.health_checker = HealthChecker()
        self.collector = get_collector()
        self.validator = ModelInputValidator()
        
        # Metrics
        self.request_counter = self.collector.counter("llm_requests_total", {"service": "llm"})
        self.latency_histogram = self.collector.histogram("llm_latency_seconds", buckets=[0.1, 0.5, 1.0, 2.5, 5.0])
        self.error_counter = self.collector.counter("llm_errors_total", {"service": "llm"})
    
    @error_handler(default_return=None)
    @timing_decorator
    def get_inference_engine(self, model_name: str) -> InferenceEngine:
        """Get or create inference engine for model."""
        if model_name in self._inference_engines:
            return self._inference_engines[model_name]
        
        # Load model
        model = self.model_manager.get_model(model_name)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create inference engine
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            use_amp=self.config.get("use_fp16", True),
            max_batch_size=self.config.get("max_batch_size", 32),
            compile_model=self.config.get("compile_model", False),
        )
        
        self._inference_engines[model_name] = engine
        
        # Emit event
        self.event_bus.publish(
            EventType.MODEL_LOADED,
            {"model_name": model_name, "device": self.device}
        )
        
        return engine
    
    @error_handler(default_return="")
    def generate_text(
        self,
        prompt: str,
        model_name: str = "gpt2",
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        user_id: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """Generate text using inference engine with enhanced features."""
        start_time = time.time()
        self.request_counter.increment()
        
        try:
            # Rate limiting
            identifier = user_id or "anonymous"
            if not self.rate_limiter.is_allowed(identifier):
                self.error_counter.increment()
                raise ValueError("Rate limit exceeded")
            
            # Sanitize input
            clean_prompt = self.sanitizer.sanitize_prompt(prompt, max_length=2048)
            
            # Validate parameters
            validate_generation_params(max_length, temperature, top_p, top_k)
            
            # Check cache
            if use_cache:
                cache_key = self.cache.make_key(
                    "generate_text",
                    clean_prompt,
                    model_name,
                    max_length,
                    temperature,
                    top_p,
                    top_k,
                )
                cached = self.cache.get(cache_key)
                if cached:
                    latency = time.time() - start_time
                    self.latency_histogram.observe(latency)
                    return cached
            
            # Get inference engine
            engine = self.get_inference_engine(model_name)
            
            # Generate (sync for now, can be made async)
            result = engine.generate(
                clean_prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                **kwargs
            )
            
            # Cache result
            if use_cache:
                self.cache.set(cache_key, result, ttl=3600)
            
            # Metrics
            latency = time.time() - start_time
            self.latency_histogram.observe(latency)
            
            return result
        
        except Exception as e:
            self.error_counter.increment()
            raise
    
    @error_handler(default_return=[])
    def get_embeddings(
        self,
        texts: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        pooling: str = "mean",
    ) -> List[List[float]]:
        """Get embeddings for texts."""
        # Get inference engine
        engine = self.get_inference_engine(model_name)
        
        # Get embeddings
        embeddings = engine.get_embeddings(
            texts,
            normalize=normalize,
            pooling=pooling,
        )
        
        return embeddings.tolist()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        try:
            info = self.service_registry.execute("model", "info", model_name=model_name)
            return info
        except Exception as e:
            return {
                "model_name": model_name,
                "loaded": False,
                "error": str(e),
            }
    
    def unload_model(self, model_name: str) -> bool:
        """Unload model from memory."""
        try:
            self.service_registry.execute("model", "unload", model_name=model_name)
            if model_name in self._inference_engines:
                del self._inference_engines[model_name]
            return True
        except Exception:
            return False
    
    async def generate_text_async(
        self,
        prompt: str,
        model_name: str = "gpt2",
        **kwargs
    ) -> str:
        """Async text generation."""
        return await self.async_executor.run(
            self.generate_text,
            prompt,
            model_name=model_name,
            **kwargs
        )
    
    async def stream_generate(
        self,
        prompt: str,
        model_name: str = "gpt2",
        max_length: int = 100,
        **kwargs
    ):
        """Stream text generation."""
        # Sanitize
        clean_prompt = self.sanitizer.sanitize_prompt(prompt)
        
        # Get engine
        engine = self.get_inference_engine(model_name)
        
        # Create streamer
        streamer = TokenStreamer(engine.model, engine.tokenizer)
        
        # Stream tokens
        async for token in streamer.stream_tokens(clean_prompt, max_length=max_length, **kwargs):
            yield token
    
    def health_check(self) -> Dict[str, Any]:
        """Get health status."""
        model = None
        if self._inference_engines:
            # Get first available model
            first_engine = next(iter(self._inference_engines.values()))
            model = first_engine.model
        
        results = self.health_checker.run_all_checks(model=model)
        overall = self.health_checker.get_overall_status(results)
        
        return {
            "status": overall,
            "components": {k: v.to_dict() for k, v in results.items()},
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get telemetry metrics."""
        return self.collector.get_all_metrics()

