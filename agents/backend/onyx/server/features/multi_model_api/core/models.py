"""
Model registry and management for multi-model API
Optimized with circuit breakers, retries, and connection pooling
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import functools

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        RetryError
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

from ..api.schemas import ModelType, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    failures: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    opened_at: Optional[datetime] = None
    
    def reset(self):
        """Reset circuit breaker"""
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"
        self.opened_at = None


@dataclass
class ModelMetadata:
    """Metadata for a model with enhanced tracking"""
    model_type: ModelType
    name: str
    description: str
    is_available: bool = True
    last_used: Optional[datetime] = None
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    call_count: int = 0
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    failure_threshold: int = 5
    recovery_timeout: int = 60
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.error_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count
    
    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency - optimized"""
        if not self.recent_latencies:
            return 0.0
        # Optimized: use list() to avoid sorting deque in-place
        sorted_latencies = sorted(self.recent_latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[index] if index < len(sorted_latencies) else sorted_latencies[-1]
    
    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows request"""
        if self.circuit_breaker.state == "closed":
            return True
        
        if self.circuit_breaker.state == "open":
            if self.circuit_breaker.opened_at:
                elapsed = (datetime.now() - self.circuit_breaker.opened_at).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.circuit_breaker.state = "half_open"
                    return True
            return False
        
        return True
    
    def record_success(self):
        """Record successful call"""
        if self.circuit_breaker.state == "half_open":
            self.circuit_breaker.reset()
        self.success_count += 1
    
    def record_failure(self):
        """Record failed call"""
        self.error_count += 1
        self.circuit_breaker.failures += 1
        self.circuit_breaker.last_failure_time = datetime.now()
        
        if self.circuit_breaker.failures >= self.failure_threshold:
            self.circuit_breaker.state = "open"
            self.circuit_breaker.opened_at = datetime.now()
            logger.warning(f"Circuit breaker opened for {self.model_type}")


class ModelRegistry:
    """Registry for AI models with circuit breakers and retries"""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
        enable_circuit_breaker: bool = True
    ):
        self.models: Dict[ModelType, ModelMetadata] = {}
        self.handlers: Dict[ModelType, Callable] = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enable_circuit_breaker = enable_circuit_breaker
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models"""
        model_configs = [
            (ModelType.COMPOSER_1, "Composer 1", "Primary composer model"),
            (ModelType.SONNET_45, "Sonnet 4.5", "Advanced sonnet model with enhanced capabilities"),
            (ModelType.GPT_51_CODEX, "GPT-5.1 Codex", "Code-focused GPT model"),
            (ModelType.GPT_51, "GPT-5.1", "Standard GPT-5.1 model"),
            (ModelType.GPT_51_CODEX_MINI, "GPT-5.1 Codex Mini", "Lightweight code model"),
            (ModelType.HAIKU_45, "Haiku 4.5", "Fast and efficient model"),
            (ModelType.GROK_CODE, "Grok Code", "Grok code generation model"),
            (ModelType.OPENROUTER_GPT4, "OpenRouter GPT-4", "GPT-4 via OpenRouter"),
            (ModelType.OPENROUTER_GPT35, "OpenRouter GPT-3.5", "GPT-3.5 Turbo via OpenRouter"),
            (ModelType.OPENROUTER_CLAUDE_OPUS, "OpenRouter Claude 3 Opus", "Claude 3 Opus via OpenRouter"),
            (ModelType.OPENROUTER_CLAUDE_SONNET, "OpenRouter Claude 3 Sonnet", "Claude 3 Sonnet via OpenRouter"),
            (ModelType.OPENROUTER_CLAUDE_HAIKU, "OpenRouter Claude 3 Haiku", "Claude 3 Haiku via OpenRouter"),
            (ModelType.OPENROUTER_GEMINI_PRO, "OpenRouter Gemini Pro", "Gemini Pro via OpenRouter"),
            (ModelType.OPENROUTER_LLAMA3_70B, "OpenRouter Llama 3 70B", "Llama 3 70B via OpenRouter"),
            (ModelType.OPENROUTER_MISTRAL_LARGE, "OpenRouter Mistral Large", "Mistral Large via OpenRouter"),
        ]
        
        for model_type, name, description in model_configs:
            self.models[model_type] = ModelMetadata(
                model_type=model_type,
                name=name,
                description=description
            )
    
    def register_handler(self, model_type: ModelType, handler: Callable):
        """Register handler for a model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found in registry")
        
        self.handlers[model_type] = handler
        logger.info(f"Registered handler for {model_type}")
    
    def get_model(self, model_type: ModelType) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self.models.get(model_type)
    
    def get_available_models(self) -> List[ModelMetadata]:
        """Get all available models"""
        return [model for model in self.models.values() if model.is_available]
    
    async def execute_model(
        self,
        model_type: ModelType,
        prompt: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ModelResponse:
        """Execute a model with a prompt - optimized with retries and circuit breaker"""
        if model_type not in self.models:
            return ModelResponse(
                model_type=model_type,
                response="",
                success=False,
                error=f"Model {model_type} not found"
            )
        
        model_meta = self.models[model_type]
        
        if not model_meta.is_available:
            return ModelResponse(
                model_type=model_type,
                response="",
                success=False,
                error=f"Model {model_type} is not available"
            )
        
        if self.enable_circuit_breaker and not model_meta.check_circuit_breaker():
            return ModelResponse(
                model_type=model_type,
                response="",
                success=False,
                error=f"Circuit breaker is open for {model_type}"
            )
        
        handler = self.handlers.get(model_type)
        
        # Optimized: avoid lambda overhead by using bound methods
        if not handler:
            if model_type.value.startswith("openrouter/"):
                handler = functools.partial(self._openrouter_handler, model_type)
            else:
                handler = functools.partial(self._mock_handler, model_type)
        
        start_time = time.time()
        model_meta.call_count += 1
        
        request_timeout = timeout or self.timeout
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if request_timeout:
                    result = await asyncio.wait_for(
                        handler(prompt, **kwargs),
                        timeout=request_timeout
                    )
                else:
                    result = await handler(prompt, **kwargs)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Optimized: batch updates with single datetime call
                now = datetime.now()
                model_meta.success_count += 1
                model_meta.total_latency_ms += latency_ms
                model_meta.recent_latencies.append(latency_ms)
                model_meta.last_used = now
                model_meta.record_success()
                
                # Optimized: single isinstance check
                if isinstance(result, dict):
                    response_text = result.get("response", "")
                    tokens = result.get("tokens_used")
                else:
                    response_text = str(result)
                    tokens = None
                
                return ModelResponse(
                    model_type=model_type,
                    response=response_text,
                    tokens_used=tokens,
                    latency_ms=latency_ms,
                    success=True
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {request_timeout}s"
                if attempt < self.max_retries - 1:
                    logger.warning(f"Timeout executing {model_type} (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    logger.error(f"Error executing {model_type} (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        latency_ms = (time.time() - start_time) * 1000
        model_meta.error_count += 1
        model_meta.total_latency_ms += latency_ms
        model_meta.record_failure()
        
        return ModelResponse(
            model_type=model_type,
            response="",
            latency_ms=latency_ms,
            success=False,
            error=last_error or "Unknown error"
        )
    
    # Optimized: class-level cache for model mapping (faster lookups)
    _model_mapping = {
        ModelType.OPENROUTER_GPT4: "openai/gpt-4",
        ModelType.OPENROUTER_GPT35: "openai/gpt-3.5-turbo",
        ModelType.OPENROUTER_CLAUDE_OPUS: "anthropic/claude-3-opus",
        ModelType.OPENROUTER_CLAUDE_SONNET: "anthropic/claude-3-sonnet",
        ModelType.OPENROUTER_CLAUDE_HAIKU: "anthropic/claude-3-haiku",
        ModelType.OPENROUTER_GEMINI_PRO: "google/gemini-pro",
        ModelType.OPENROUTER_LLAMA3_70B: "meta-llama/llama-3-70b-instruct",
        ModelType.OPENROUTER_MISTRAL_LARGE: "mistralai/mistral-large",
    }
    
    async def _openrouter_handler(self, model_type: ModelType, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handler for OpenRouter models - ultra optimized"""
        try:
            from ..integrations.openrouter import openrouter_handler
            
            # Optimized: use direct dict access with fallback chain
            openrouter_model = (
                kwargs.get("openrouter_model") or 
                (kwargs.get("custom_params") or {}).get("openrouter_model") or
                self._model_mapping.get(model_type, "openai/gpt-3.5-turbo")
            )
            
            # Optimized: build filtered kwargs more efficiently
            custom_params = kwargs.get("custom_params")
            exclude_keys = {"openrouter_model", "custom_params"}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}
            
            if custom_params:
                filtered_kwargs.update({k: v for k, v in custom_params.items() if k != "openrouter_model"})
            
            # Optimized: extract common params once
            temperature = kwargs.get("temperature")
            max_tokens = kwargs.get("max_tokens")
            
            result = await openrouter_handler(
                prompt=prompt,
                model=openrouter_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **filtered_kwargs
            )
            
            return result
        except ImportError:
            logger.warning("OpenRouter integration not available. Install httpx to use OpenRouter models.")
            return await self._mock_handler(model_type, prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error calling OpenRouter for {model_type}: {e}")
            raise
    
    async def _mock_handler(self, model_type: ModelType, prompt: str, **kwargs) -> Dict[str, Any]:
        """Mock handler for development/testing"""
        await asyncio.sleep(0.1)
        return {
            "response": f"Mock response from {model_type.value} for prompt: {prompt[:50]}...",
            "tokens_used": len(prompt.split()) + 50
        }
    
    async def execute_models_parallel(
        self,
        model_configs: List[Tuple[ModelType, str, dict]],
        max_concurrent: int = 5
    ) -> List[ModelResponse]:
        """Execute multiple models in parallel with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(model_type, prompt, kwargs):
            async with semaphore:
                return await self.execute_model(model_type, prompt, **kwargs)
        
        tasks = [
            execute_with_semaphore(model_type, prompt, kwargs)
            for model_type, prompt, kwargs in model_configs
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def get_model_health(self, model_type: ModelType) -> Dict[str, Any]:
        """Get health metrics for a model"""
        if model_type not in self.models:
            return {"error": "Model not found"}
        
        model_meta = self.models[model_type]
        return {
            "model_type": model_type.value,
            "is_available": model_meta.is_available,
            "success_rate": round(model_meta.success_rate, 2),
            "avg_latency_ms": round(model_meta.avg_latency_ms, 2),
            "p95_latency_ms": round(model_meta.p95_latency_ms, 2),
            "call_count": model_meta.call_count,
            "circuit_breaker_state": model_meta.circuit_breaker.state,
            "circuit_breaker_failures": model_meta.circuit_breaker.failures,
            "last_used": model_meta.last_used.isoformat() if model_meta.last_used else None
        }
    
    def update_availability(self, model_type: ModelType, is_available: bool):
        """Update model availability"""
        if model_type in self.models:
            self.models[model_type].is_available = is_available
            logger.info(f"Model {model_type} availability set to {is_available}")


# Global registry instance
_registry_instance: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance

