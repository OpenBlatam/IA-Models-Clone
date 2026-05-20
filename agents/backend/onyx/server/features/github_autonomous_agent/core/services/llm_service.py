"""
LLM Service - Servicio mejorado para integración con modelos de IA vía OpenRouter.

Características:
- Caché inteligente de respuestas
- Retry logic con exponential backoff
- Streaming support
- Métricas y estadísticas
- Validación de inputs
- Circuit breaker
- Rate limiting por modelo
- Token counting/estimation
- Batch processing
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

from config.logging_config import get_logger
from config.settings import settings
from core.exceptions import TaskProcessingError
from core.services.cache_service import CacheService

# Importar componentes modulares
from core.services.llm import (
    get_template_registry,
    get_token_manager,
    get_validator,
    get_model_registry,
    get_model_selector,
    get_cost_optimizer,
    get_profiler,
    get_data_pipeline,
    DataPipeline,
    get_checkpoint_manager,
    get_llm_analytics,
    get_performance_optimizer,
    MetricType,
    ValidationLevel,
    TokenInfo,
    SelectionStrategy,
    SelectionCriteria
)

logger = get_logger(__name__)

# Import opcional de MetricsService
try:
    from core.services.metrics_service import MetricsService
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    MetricsService = None


class FinishReason(str, Enum):
    """Razones de finalización de generación."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass
class LLMResponse:
    """Respuesta de un modelo LLM."""
    model: str
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    cached: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)


@dataclass
class LLMRequest:
    """Solicitud a un modelo LLM."""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stream: bool = False
    cache_enabled: bool = True
    cache_ttl: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para payload."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.top_p:
            payload["top_p"] = self.top_p
        if self.frequency_penalty:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            payload["presence_penalty"] = self.presence_penalty
        if self.stream:
            payload["stream"] = self.stream
        
        return payload


@dataclass
class LLMStats:
    """Estadísticas del servicio LLM."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    requests_by_model: Dict[str, int] = None
    errors_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.requests_by_model is None:
            self.requests_by_model = defaultdict(int)
        if self.errors_by_type is None:
            self.errors_by_type = defaultdict(int)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "total_tokens_prompt": self.total_tokens_prompt,
            "total_tokens_completion": self.total_tokens_completion,
            "total_tokens": self.total_tokens,
            "average_latency_ms": self.average_latency_ms,
            "requests_by_model": dict(self.requests_by_model),
            "errors_by_type": dict(self.errors_by_type),
        }


class CircuitBreaker:
    """Circuit breaker simple para evitar sobrecargar la API."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self):
        """Registrar éxito, resetear contador."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Registrar fallo."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker abierto después de {self.failure_count} fallos")
    
    def can_attempt(self) -> bool:
        """Verificar si se puede intentar una request."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker en estado half-open, intentando recuperación")
                    return True
            return False
        
        # half_open
        return True


class LLMService:
    """
    Servicio mejorado para interactuar con modelos de IA a través de OpenRouter.
    
    Características:
    - Caché inteligente de respuestas
    - Retry logic con exponential backoff
    - Streaming support
    - Métricas y estadísticas detalladas
    - Validación de inputs
    - Circuit breaker
    - Rate limiting por modelo
    """
    
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    
    # Estimación aproximada: 1 token ≈ 4 caracteres
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_models: Optional[List[str]] = None,
        timeout: int = 60,
        max_parallel_requests: int = 10,
        cache_service: Optional[CacheService] = None,
        metrics_service: Optional['MetricsService'] = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hora por defecto
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        circuit_breaker_failures: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        """
        Inicializar servicio LLM mejorado.
        
        Args:
            api_key: API key de OpenRouter
            default_models: Lista de modelos por defecto
            timeout: Timeout en segundos para requests
            max_parallel_requests: Máximo de requests paralelos
            cache_service: Servicio de caché (opcional)
            enable_cache: Habilitar caché de respuestas
            cache_ttl: TTL de caché en segundos
            max_retries: Número máximo de reintentos
            retry_min_wait: Tiempo mínimo de espera entre reintentos
            retry_max_wait: Tiempo máximo de espera entre reintentos
            circuit_breaker_failures: Fallos antes de abrir circuit breaker
            circuit_breaker_timeout: Tiempo de recuperación del circuit breaker
        """
        self.api_key = api_key or getattr(settings, 'OPENROUTER_API_KEY', '')
        if not self.api_key:
            logger.warning("OpenRouter API key no configurada. Algunas funciones pueden no funcionar.")
        
        self.default_models = default_models or getattr(
            settings, 
            'LLM_DEFAULT_MODELS', 
            [
                'openai/gpt-4o-mini',
                'anthropic/claude-3.5-sonnet',
                'google/gemini-pro-1.5'
            ]
        )
        
        self.timeout = timeout
        self.max_parallel_requests = max_parallel_requests
        self.semaphore = asyncio.Semaphore(max_parallel_requests)
        
        # Caché
        self.cache_service = cache_service
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Métricas
        self.metrics_service = metrics_service
        
        # Componentes modulares
        self.template_registry = get_template_registry()
        self.token_manager = get_token_manager()
        self.validator = get_validator(ValidationLevel.BASIC)
        self.model_registry = get_model_registry()
        self.model_selector = get_model_selector()
        self.cost_optimizer = get_cost_optimizer()
        self.profiler = get_profiler()
        self.data_pipeline = get_data_pipeline()
        self.checkpoint_manager = get_checkpoint_manager()
        self.analytics = get_llm_analytics()
        self.performance_optimizer = get_performance_optimizer()
        
        # Retry config
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_failures,
            recovery_timeout=circuit_breaker_timeout
        )
        
        # Estadísticas
        self.stats = LLMStats()
        
        # Rate limiting por modelo (simple in-memory)
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.rate_limit_window = 60  # 1 minuto
        self.rate_limit_max = 50  # 50 requests por minuto por modelo
        
        # Remover constante antigua (ahora se usa TokenManager)
        # self.CHARS_PER_TOKEN ya no se usa directamente
        
        # Cliente HTTP async con connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.OPENROUTER_API_BASE,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100
            ),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": getattr(settings, 'OPENROUTER_HTTP_REFERER', ''),
                "X-Title": getattr(settings, 'OPENROUTER_X_TITLE', 'GitHub Autonomous Agent'),
                "Content-Type": "application/json",
            } if self.api_key else {}
        )
        
        logger.info(
            f"LLM Service inicializado con {len(self.default_models)} modelos por defecto, "
            f"caché={'habilitado' if enable_cache else 'deshabilitado'}"
        )
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generar clave de caché basada en el request."""
        # Crear hash del request (sin campos que no afectan la respuesta)
        key_data = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return f"llm:{request.model}:{key_hash[:16]}"
    
    def _estimate_tokens(self, text: str, content_type: str = "text") -> int:
        """Estimar número de tokens en un texto usando TokenManager."""
        return self.token_manager.estimate_tokens(text, content_type)
    
    def _validate_request(self, request: LLMRequest) -> None:
        """Validar request antes de enviarlo usando TokenManager."""
        if not request.model:
            raise ValueError("Modelo no especificado")
        
        if not request.messages:
            raise ValueError("Messages no puede estar vacío")
        
        if not (0.0 <= request.temperature <= 2.0):
            raise ValueError("Temperature debe estar entre 0.0 y 2.0")
        
        if request.max_tokens and request.max_tokens < 1:
            raise ValueError("max_tokens debe ser mayor a 0")
        
        # Validar usando TokenManager
        is_valid, error_msg, token_info = self.token_manager.validate_request(
            request.model,
            request.messages,
            request.max_tokens
        )
        
        if not is_valid:
            raise ValueError(error_msg or "Validación de request falló")
        
        # Registrar estimación de tokens
        self.token_manager.record_usage(request.model, token_info)
    
    def _check_rate_limit(self, model: str) -> bool:
        """Verificar rate limit para un modelo."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.rate_limit_window)
        
        # Limpiar requests antiguos
        self.rate_limits[model] = [
            req_time for req_time in self.rate_limits[model]
            if req_time > cutoff
        ]
        
        # Verificar límite
        if len(self.rate_limits[model]) >= self.rate_limit_max:
            logger.warning(f"Rate limit alcanzado para modelo {model}")
            return False
        
        # Registrar request
        self.rate_limits[model].append(now)
        return True
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generar respuesta de un modelo LLM con caché y retry.
        
        Args:
            prompt: Prompt del usuario
            model: Modelo a usar
            system_prompt: Prompt del sistema (opcional)
            temperature: Temperatura para sampling
            max_tokens: Máximo de tokens a generar
            cache_enabled: Habilitar caché para este request (sobrescribe configuración global)
            **kwargs: Parámetros adicionales
            
        Returns:
            LLMResponse con la respuesta del modelo
        """
        if not self.api_key:
            raise TaskProcessingError("OpenRouter API key no configurada")
        
        model = model or self.default_models[0]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request = LLMRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_enabled=cache_enabled if cache_enabled is not None else self.enable_cache,
            **kwargs
        )
        
        return await self._process_request(request)
    
    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generar respuesta en streaming.
        
        Args:
            prompt: Prompt del usuario
            model: Modelo a usar
            system_prompt: Prompt del sistema
            temperature: Temperatura
            max_tokens: Máximo de tokens
            **kwargs: Parámetros adicionales
            
        Yields:
            Chunks de texto conforme se generan
        """
        if not self.api_key:
            raise TaskProcessingError("OpenRouter API key no configurada")
        
        model = model or self.default_models[0]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request = LLMRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            cache_enabled=False,  # No cachear streaming
            **kwargs
        )
        
        async for chunk in self._process_stream(request):
            yield chunk
    
    async def generate_parallel(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, LLMResponse]:
        """
        Generar respuestas de múltiples modelos en paralelo.
        
        Args:
            prompt: Prompt del usuario
            models: Lista de modelos a usar
            system_prompt: Prompt del sistema
            temperature: Temperatura
            max_tokens: Máximo de tokens
            **kwargs: Parámetros adicionales
            
        Returns:
            Diccionario con modelo como clave y LLMResponse como valor
        """
        if not self.api_key:
            raise TaskProcessingError("OpenRouter API key no configurada")
        
        models = models or self.default_models
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Crear requests para todos los modelos
        requests = [
            LLMRequest(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            for model in models
        ]
        
        # Ejecutar en paralelo
        start_time = datetime.now()
        tasks = [self._process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        results = {}
        for model, response in zip(models, responses):
            if isinstance(response, Exception):
                logger.error(f"Error al generar respuesta con modelo {model}: {response}")
                results[model] = LLMResponse(
                    model=model,
                    content="",
                    error=str(response)
                )
            else:
                results[model] = response
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Generadas {len(results)} respuestas en paralelo en {total_time:.2f}ms")
        
        return results
    
    async def _process_request(self, request: LLMRequest) -> LLMResponse:
        """Procesar request con caché y retry."""
        self.stats.total_requests += 1
        self.stats.requests_by_model[request.model] += 1
        
        # Iniciar profiling
        self.profiler.start(f"llm_request_{request.model}")
        
        # Iniciar timer de métricas
        if self.metrics_service:
            self.metrics_service.start_timer(f"llm_request_{request.model}")
        
        # Validar request
        try:
            self._validate_request(request)
        except ValueError as e:
            self.stats.failed_requests += 1
            self.stats.errors_by_type["validation_error"] += 1
            return LLMResponse(
                model=request.model,
                content="",
                error=f"Error de validación: {str(e)}"
            )
        
        # Verificar rate limit
        if not self._check_rate_limit(request.model):
            self.stats.failed_requests += 1
            self.stats.errors_by_type["rate_limit"] += 1
            return LLMResponse(
                model=request.model,
                content="",
                error="Rate limit alcanzado para este modelo"
            )
        
        # Verificar caché
        if request.cache_enabled and self.cache_service:
            cache_key = self._generate_cache_key(request)
            cached_response = self.cache_service.get(cache_key)
            
            if cached_response:
                self.stats.cache_hits += 1
                logger.debug(f"Cache hit para modelo {request.model}")
                
                if self.metrics_service:
                    self.metrics_service.record_cache_operation("get", "hit")
                
                # Convertir respuesta cached a LLMResponse
                if isinstance(cached_response, dict):
                    response = LLMResponse(**cached_response)
                    response.cached = True
                    return response
        
        if request.cache_enabled:
            self.stats.cache_misses += 1
            if self.metrics_service:
                self.metrics_service.record_cache_operation("get", "miss")
        
        # Verificar circuit breaker
        if not self.circuit_breaker.can_attempt():
            self.stats.failed_requests += 1
            self.stats.errors_by_type["circuit_breaker"] += 1
            return LLMResponse(
                model=request.model,
                content="",
                error="Circuit breaker abierto, servicio temporalmente no disponible"
            )
        
        # Procesar request con retry
        retry_count = 0
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self._make_request(request)
                
                if response.error:
                    last_error = response.error
                    retry_count += 1
                    
                    # Algunos errores no deben reintentarse
                    if "401" in response.error or "403" in response.error:
                        break
                    
                    if attempt < self.max_retries - 1:
                        wait_time = min(
                            self.retry_min_wait * (2 ** attempt),
                            self.retry_max_wait
                        )
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    # Éxito
                    self.circuit_breaker.record_success()
                    self.stats.successful_requests += 1
                    response.retry_count = retry_count
                    
                    # Registrar métricas de éxito
                    duration = self.profiler.stop(
                        f"llm_request_{request.model}",
                        tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                        cost=getattr(response, '_estimated_cost', 0.0)
                    )
                    
                    if self.metrics_service:
                        if duration:
                            self.metrics_service.record_api_request(
                                f"llm_{request.model}",
                                "success",
                                duration
                            )
                    
                    # Registrar en analytics y performance optimizer
                    if response.latency_ms:
                        self.analytics.record_metric(
                            MetricType.LATENCY,
                            response.latency_ms,
                            tags={"model": request.model, "status": "success"}
                        )
                        
                        # Registrar en performance optimizer
                        self.performance_optimizer.record_metrics(
                            latency_ms=response.latency_ms,
                            error=False,
                            cached=response.cached
                        )
                    
                    if response.usage:
                        total_tokens = response.usage.get("total_tokens", 0)
                        if total_tokens > 0:
                            self.analytics.record_metric(
                                MetricType.TOKEN_USAGE,
                                total_tokens,
                                tags={"model": request.model}
                            )
                    
                    # Registrar request count
                    self.analytics.record_metric(
                        MetricType.REQUEST_COUNT,
                        1.0,
                        tags={"model": request.model, "status": "success"}
                    )
                    
                    # Registrar cache hit si aplica
                    if response.cached:
                        self.analytics.record_metric(
                            MetricType.CACHE_HIT_RATE,
                            1.0,
                            tags={"model": request.model}
                        )
                    else:
                        self.analytics.record_metric(
                            MetricType.CACHE_HIT_RATE,
                            0.0,
                            tags={"model": request.model}
                        )
                    
                    # Guardar en caché
                    if request.cache_enabled and self.cache_service and not response.error:
                        cache_key = self._generate_cache_key(request)
                        self.cache_service.set(
                            cache_key,
                            response.to_dict(),
                            ttl=request.cache_ttl or self.cache_ttl
                        )
                        if self.metrics_service:
                            self.metrics_service.record_cache_operation("set", "success")
                    
                    return response
                    
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                logger.warning(f"Error en intento {attempt + 1}/{self.max_retries}: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = min(
                        self.retry_min_wait * (2 ** attempt),
                        self.retry_max_wait
                    )
                    await asyncio.sleep(wait_time)
        
        # Todos los intentos fallaron
        self.circuit_breaker.record_failure()
        self.stats.failed_requests += 1
        self.stats.errors_by_type["request_failed"] += 1
        
        # Registrar métricas de error
        duration = self.profiler.stop(f"llm_request_{request.model}", error=True)
        
        if self.metrics_service:
            if duration:
                self.metrics_service.record_api_request(
                    f"llm_{request.model}",
                    "error",
                    duration
                )
            self.metrics_service.record_error("LLMRequestFailed")
        
        # Registrar en analytics
        self.analytics.record_metric(
            MetricType.REQUEST_COUNT,
            1.0,
            tags={"model": request.model, "status": "failed"}
        )
        
        self.analytics.record_metric(
            MetricType.ERROR_RATE,
            1.0,
            tags={"model": request.model, "error_type": "request_failed"}
        )
        
        if duration:
            self.analytics.record_metric(
                MetricType.LATENCY,
                duration * 1000,  # Convertir a ms
                tags={"model": request.model, "status": "failed"}
            )
            
            # Registrar en performance optimizer
            self.performance_optimizer.record_metrics(
                latency_ms=duration * 1000,
                error=True,
                cached=False
            )
        
        return LLMResponse(
            model=request.model,
            content="",
            error=f"Error después de {retry_count} intentos: {last_error}",
            retry_count=retry_count
        )
    
    async def _process_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Procesar request en modo streaming."""
        async with self.semaphore:
            try:
                payload = request.to_dict()
                
                async with self.client.stream(
                    "POST",
                    "/chat/completions",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                logger.error(f"Error en streaming: {e}")
                raise
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Hacer request a OpenRouter API."""
        async with self.semaphore:
            start_time = datetime.now()
            
            try:
                payload = request.to_dict()
                
                response = await self.client.post(
                    "/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Actualizar estadísticas
                self.stats.total_latency_ms += latency_ms
                if self.stats.total_requests > 0:
                    self.stats.average_latency_ms = (
                        self.stats.total_latency_ms / self.stats.total_requests
                    )
                
                # Extraer respuesta
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                # Actualizar estadísticas de tokens
                usage = data.get("usage", {})
                if usage:
                    self.stats.total_tokens_prompt += usage.get("prompt_tokens", 0)
                    self.stats.total_tokens_completion += usage.get("completion_tokens", 0)
                    self.stats.total_tokens += usage.get("total_tokens", 0)
                
                response = LLMResponse(
                    model=request.model,
                    content=content,
                    usage=usage,
                    finish_reason=choice.get("finish_reason"),
                    latency_ms=latency_ms
                )
                
                # Actualizar token info con valores reales
                if usage:
                    token_info = TokenInfo(
                        estimated_tokens=self.token_manager.estimate_messages_tokens(request.messages),
                        actual_tokens=usage.get("total_tokens"),
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens")
                    )
                    self.token_manager.record_usage(request.model, token_info)
                    
                    # Registrar costo
                    cost, within_budget = self.cost_optimizer.record_cost(
                        request.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        use_case=getattr(request, 'use_case', None)
                    )
                    
                    # Registrar costo en analytics
                    if cost > 0:
                        self.analytics.record_metric(
                            MetricType.COST,
                            cost,
                            tags={"model": request.model}
                        )
                    
                    if not within_budget:
                        logger.warning(f"Request excede presupuesto: ${cost:.4f}")
                        # Registrar alerta de presupuesto
                        self.analytics.record_metric(
                            MetricType.COST,
                            cost,
                            tags={"model": request.model, "exceeded_budget": "true"}
                        )
                
                return response
                
            except httpx.HTTPStatusError as e:
                error_msg = f"Error HTTP {e.response.status_code}: {e.response.text}"
                logger.error(f"Error en request a OpenRouter: {error_msg}")
                return LLMResponse(
                    model=request.model,
                    content="",
                    error=error_msg
                )
            except httpx.RequestError as e:
                error_msg = f"Error de conexión: {str(e)}"
                logger.error(f"Error de conexión a OpenRouter: {error_msg}")
                return LLMResponse(
                    model=request.model,
                    content="",
                    error=error_msg
                )
            except Exception as e:
                error_msg = f"Error inesperado: {str(e)}"
                logger.error(f"Error inesperado en LLM request: {error_msg}", exc_info=True)
                return LLMResponse(
                    model=request.model,
                    content="",
                    error=error_msg
                )
    
    async def analyze_code(
        self,
        code: str,
        language: Optional[str] = None,
        analysis_type: str = "general",
        model: Optional[str] = None,
        validate_response: bool = True
    ) -> LLMResponse:
        """
        Analizar código usando un modelo LLM con templates.
        
        Args:
            code: Código a analizar
            language: Lenguaje de programación (opcional, se detecta automáticamente)
            analysis_type: Tipo de análisis (general, bugs, performance, security, style)
            model: Modelo a usar (opcional)
            validate_response: Validar respuesta antes de retornar
            
        Returns:
            LLMResponse con el análisis
        """
        # Usar template registry
        template = self.template_registry.get("code_analysis")
        if template:
            prompts = template.render(
                code=code,
                language=f" de {language}" if language else "",
                analysis_type=analysis_type
            )
            system_prompt = prompts["system_prompt"]
            user_prompt = prompts["user_prompt"]
        else:
            # Fallback a implementación anterior
            system_prompts = {
                "general": "Eres un experto analista de código. Analiza el código proporcionado y proporciona feedback constructivo sobre estructura, calidad y mejores prácticas.",
                "bugs": "Eres un experto en detección de bugs. Analiza el código y identifica posibles errores, bugs, edge cases o problemas lógicos.",
                "performance": "Eres un experto en optimización de código. Analiza el código y sugiere mejoras de rendimiento, optimizaciones y mejores prácticas de performance.",
                "security": "Eres un experto en seguridad de código. Analiza el código y identifica vulnerabilidades de seguridad, problemas de autenticación, autorización y mejores prácticas de seguridad.",
                "style": "Eres un experto en estilo de código. Analiza el código y sugiere mejoras de estilo, legibilidad, naming conventions y estructura."
            }
            system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
            user_prompt = f"""Analiza el siguiente código{' de ' + language if language else ''}:

```{language or ''}
{code}
```

Proporciona un análisis detallado enfocado en: {analysis_type}. Incluye:
- Problemas identificados
- Sugerencias de mejora
- Ejemplos de código mejorado si es relevante"""
        
        response = await self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=template.default_temperature if template else 0.3
        )
        
        # Validar respuesta si está habilitado
        if validate_response and not response.error:
            validation = self.validator.validate(
                response.content,
                expected_format="markdown",
                min_length=100
            )
            if not validation.is_valid:
                logger.warning(f"Respuesta de análisis con problemas: {validation.issues}")
        
        return response
    
    async def generate_instruction(
        self,
        description: str,
        context: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Generar instrucción estructurada a partir de una descripción en lenguaje natural.
        
        Args:
            description: Descripción de lo que se quiere hacer
            context: Contexto adicional (código, archivos, etc.)
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con la instrucción generada
        """
        system_prompt = """Eres un asistente que convierte descripciones en lenguaje natural en instrucciones estructuradas para un agente autónomo de GitHub.

Las instrucciones deben ser claras, específicas y ejecutables. Formato preferido:
- create file: path/to/file.ext
- update file: path/to/file.ext
- create branch: branch-name
- create pr: title, body, head, base

Responde solo con la instrucción, sin explicaciones adicionales."""
        
        prompt = f"Descripción: {description}"
        if context:
            prompt += f"\n\nContexto:\n{context}"
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.5
        )
    
    async def generate_documentation(
        self,
        code: str,
        language: Optional[str] = None,
        doc_type: str = "function",
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Generar documentación para código.
        
        Args:
            code: Código a documentar
            language: Lenguaje de programación
            doc_type: Tipo de documentación (function, class, module, api)
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con la documentación generada
        """
        doc_prompts = {
            "function": "Genera documentación completa para esta función incluyendo: descripción, parámetros, retorno, ejemplos y excepciones.",
            "class": "Genera documentación completa para esta clase incluyendo: descripción, atributos, métodos y ejemplos de uso.",
            "module": "Genera documentación completa para este módulo incluyendo: descripción general, funciones principales, clases y ejemplos.",
            "api": "Genera documentación de API REST para este código incluyendo: endpoints, parámetros, respuestas y ejemplos."
        }
        
        system_prompt = f"""Eres un experto en documentación de código. {doc_prompts.get(doc_type, doc_prompts["function"])}

Sigue las convenciones del lenguaje {language or 'especificado'} y genera documentación clara, concisa y completa."""
        
        prompt = f"""Genera documentación para el siguiente código:

```{language or ''}
{code}
```"""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.3
        )
    
    async def refactor_code(
        self,
        code: str,
        language: Optional[str] = None,
        refactor_type: str = "general",
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Refactorizar código usando un modelo LLM.
        
        Args:
            code: Código a refactorizar
            language: Lenguaje de programación
            refactor_type: Tipo de refactorización (general, performance, readability, patterns)
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con el código refactorizado
        """
        refactor_prompts = {
            "general": "Refactoriza el código mejorando su estructura, legibilidad y mantenibilidad sin cambiar su funcionalidad.",
            "performance": "Refactoriza el código optimizando su rendimiento, reduciendo complejidad temporal y espacial.",
            "readability": "Refactoriza el código mejorando su legibilidad, nombres de variables, estructura y comentarios.",
            "patterns": "Refactoriza el código aplicando patrones de diseño apropiados y mejores prácticas del lenguaje."
        }
        
        system_prompt = f"""Eres un experto en refactorización de código. {refactor_prompts.get(refactor_type, refactor_prompts["general"])}

Mantén la funcionalidad original pero mejora la calidad del código. Incluye una explicación breve de los cambios realizados."""
        
        prompt = f"""Refactoriza el siguiente código{' de ' + language if language else ''}:

```{language or ''}
{code}
```

Proporciona el código refactorizado y una explicación de los cambios."""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.4
        )
    
    async def generate_tests(
        self,
        code: str,
        language: Optional[str] = None,
        test_framework: Optional[str] = None,
        test_type: str = "unit",
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Generar tests para código.
        
        Args:
            code: Código para el cual generar tests
            language: Lenguaje de programación
            test_framework: Framework de testing (pytest, unittest, jest, etc.)
            test_type: Tipo de test (unit, integration, e2e)
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con los tests generados
        """
        test_prompts = {
            "unit": "Genera tests unitarios completos que cubran casos normales, edge cases y casos de error.",
            "integration": "Genera tests de integración que verifiquen la interacción entre componentes.",
            "e2e": "Genera tests end-to-end que verifiquen flujos completos de la aplicación."
        }
        
        framework_hints = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "java": "junit",
            "go": "testing"
        }
        
        if not test_framework and language:
            test_framework = framework_hints.get(language.lower(), "estándar del lenguaje")
        
        system_prompt = f"""Eres un experto en testing. {test_prompts.get(test_type, test_prompts["unit"])}

Usa el framework {test_framework or 'estándar'} y sigue las mejores prácticas de testing. Incluye casos de prueba comprehensivos."""
        
        prompt = f"""Genera tests {test_type} para el siguiente código{' de ' + language if language else ''}:

```{language or ''}
{code}
```

Proporciona tests completos y bien estructurados."""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.3
        )
    
    async def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        detail_level: str = "medium",
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Explicar código en lenguaje natural.
        
        Args:
            code: Código a explicar
            language: Lenguaje de programación
            detail_level: Nivel de detalle (simple, medium, detailed)
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con la explicación
        """
        detail_prompts = {
            "simple": "Explica el código de forma simple y concisa, como si explicaras a un principiante.",
            "medium": "Explica el código de forma clara, incluyendo qué hace, cómo funciona y los conceptos clave.",
            "detailed": "Explica el código en detalle, línea por línea si es necesario, incluyendo conceptos avanzados y contexto."
        }
        
        system_prompt = f"""Eres un experto en programación. {detail_prompts.get(detail_level, detail_prompts["medium"])}

Proporciona una explicación clara y educativa del código."""
        
        prompt = f"""Explica el siguiente código{' de ' + language if language else ''}:

```{language or ''}
{code}
```"""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.5
        )
    
    async def suggest_improvements(
        self,
        code: str,
        language: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Sugerir mejoras para código.
        
        Args:
            code: Código a mejorar
            language: Lenguaje de programación
            focus_areas: Áreas de enfoque (performance, security, readability, maintainability)
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con sugerencias de mejora
        """
        focus_areas = focus_areas or ["general"]
        
        system_prompt = """Eres un experto en revisión de código. Analiza el código y proporciona sugerencias de mejora específicas y accionables.

Organiza las sugerencias por categoría y prioridad. Incluye ejemplos de código mejorado cuando sea relevante."""
        
        focus_text = ", ".join(focus_areas) if focus_areas else "general"
        
        prompt = f"""Analiza el siguiente código{' de ' + language if language else ''} y sugiere mejoras enfocadas en: {focus_text}

```{language or ''}
{code}
```

Proporciona sugerencias específicas, priorizadas y con ejemplos cuando sea posible."""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.4
        )
    
    async def compare_code_versions(
        self,
        code_before: str,
        code_after: str,
        language: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Comparar dos versiones de código y explicar las diferencias.
        
        Args:
            code_before: Versión anterior del código
            code_after: Versión nueva del código
            language: Lenguaje de programación
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con la comparación
        """
        system_prompt = """Eres un experto en revisión de código. Compara dos versiones de código y explica:
- Las diferencias principales
- Mejoras o regresiones
- Impacto en funcionalidad, rendimiento y mantenibilidad
- Recomendaciones"""
        
        prompt = f"""Compara estas dos versiones de código{' de ' + language if language else ''}:

**Versión Anterior:**
```{language or ''}
{code_before}
```

**Versión Nueva:**
```{language or ''}
{code_after}
```

Proporciona un análisis detallado de las diferencias y su impacto."""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.3
        )
    
    async def generate_code_from_description(
        self,
        description: str,
        language: str,
        requirements: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Generar código a partir de una descripción en lenguaje natural.
        
        Args:
            description: Descripción de lo que debe hacer el código
            language: Lenguaje de programación
            requirements: Requisitos adicionales (opcional)
            model: Modelo a usar (opcional)
            
        Returns:
            LLMResponse con el código generado
        """
        system_prompt = f"""Eres un experto programador en {language}. Genera código limpio, bien estructurado y siguiendo las mejores prácticas del lenguaje.

El código debe ser:
- Funcional y correcto
- Bien documentado
- Siguiendo convenciones del lenguaje
- Incluyendo manejo de errores apropiado"""
        
        prompt = f"""Genera código en {language} que:

{description}"""
        
        if requirements:
            prompt += f"\n\nRequisitos adicionales:\n{requirements}"
        
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.5
        )
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Obtener lista de modelos disponibles en OpenRouter."""
        try:
            self.profiler.start("get_available_models")
            
            if self.metrics_service:
                self.metrics_service.start_timer("llm_list_models")
            
            response = await self.client.get("/models")
            response.raise_for_status()
            data = response.json()
            
            duration = self.profiler.stop("get_available_models")
            
            if self.metrics_service:
                self.metrics_service.stop_timer("llm_list_models")
                self.metrics_service.record_api_request("llm_list_models", "success", duration)
            
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error al obtener modelos disponibles: {e}")
            self.profiler.stop("get_available_models", error=True)
            if self.metrics_service:
                self.metrics_service.record_error("LLMListModelsError")
            return []
    
    def select_optimal_model(
        self,
        use_case: str,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        max_cost: Optional[float] = None
    ) -> Optional[str]:
        """
        Seleccionar modelo óptimo para un caso de uso.
        
        Args:
            use_case: Caso de uso
            strategy: Estrategia de selección
            max_cost: Costo máximo por request
            
        Returns:
            ID del modelo seleccionado o None
        """
        criteria = SelectionCriteria(
            use_case=use_case,
            strategy=strategy,
            max_cost=max_cost
        )
        
        model_config = self.model_selector.select_model(criteria)
        return model_config.model_id if model_config else None
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de costos."""
        return self.cost_optimizer.get_stats()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte de performance."""
        return self.profiler.get_summary()
    
    def save_checkpoint(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Guardar checkpoint del estado actual.
        
        Args:
            name: Nombre del checkpoint
            metadata: Metadatos adicionales
            
        Returns:
            Ruta del checkpoint
        """
        state = {
            "stats": self.stats.to_dict(),
            "default_models": self.default_models,
            "config": {
                "timeout": self.timeout,
                "max_parallel_requests": self.max_parallel_requests,
                "enable_cache": self.enable_cache
            }
        }
        
        return self.checkpoint_manager.save_checkpoint(name, state, metadata)
    
    def process_code_with_pipeline(
        self,
        code: str
    ) -> str:
        """
        Procesar código usando el pipeline de datos.
        
        Args:
            code: Código a procesar
            
        Returns:
            Código procesado
        """
        pipeline = DataPipeline.create_code_pipeline()
        processed = pipeline.process(code)
        return processed.processed
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio."""
        return self.stats.to_dict()
    
    def reset_stats(self) -> None:
        """Resetear estadísticas."""
        self.stats = LLMStats()
        logger.info("Estadísticas del servicio LLM reseteadas")
    
    async def close(self):
        """Cerrar cliente HTTP y limpiar recursos."""
        await self.client.aclose()
        logger.info("LLM Service cerrado")
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
