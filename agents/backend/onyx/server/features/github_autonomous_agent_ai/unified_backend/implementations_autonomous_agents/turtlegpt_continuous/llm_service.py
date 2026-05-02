"""
LLM Service Module
==================

Servicio encapsulado para llamadas LLM con manejo de métricas,
retry, caching y otros aspectos comunes.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .prompt_builder import PromptBuilder
from .error_handler import LLMError, ErrorRecoveryStrategy
from .decorators import async_retry, async_timeout
from .constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_RETRY_COUNT

logger = logging.getLogger(__name__)


class LLMService:
    """
    Servicio encapsulado para llamadas LLM.
    
    Proporciona:
    - Manejo automático de métricas
    - Retry con backoff
    - Timeout handling
    - Caching opcional
    - Tracking de uso
    """
    
    def __init__(
        self,
        llm_client,
        metrics_manager=None,
        enable_retry: bool = True,
        enable_timeout: bool = True,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        default_temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Inicializar servicio LLM.
        
        Args:
            llm_client: Cliente LLM subyacente
            metrics_manager: Manager de métricas
            enable_retry: Habilitar retry automático
            enable_timeout: Habilitar timeout
            default_max_tokens: Tokens máximos por defecto
            default_temperature: Temperatura por defecto
        """
        self.llm_client = llm_client
        self.metrics_manager = metrics_manager
        self.enable_retry = enable_retry
        self.enable_timeout = enable_timeout
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        
        # Estadísticas
        self.total_calls = 0
        self.total_tokens = 0
        self.total_errors = 0
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_retry: Optional[bool] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generar texto usando LLM con manejo completo.
        
        Args:
            prompt: Prompt a enviar
            max_tokens: Tokens máximos (None = default)
            temperature: Temperatura (None = default)
            use_retry: Usar retry (None = usar self.enable_retry)
            timeout: Timeout en segundos (None = sin timeout)
            **kwargs: Argumentos adicionales para el cliente LLM
            
        Returns:
            Respuesta del LLM con métricas
        """
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        use_retry = use_retry if use_retry is not None else self.enable_retry
        
        start_time = datetime.now()
        
        try:
            # Función interna para la llamada
            async def _call_llm():
                response = await self.llm_client.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response
            
            # Aplicar retry si está habilitado
            if use_retry:
                _call_llm = async_retry(
                    max_retries=DEFAULT_RETRY_COUNT,
                    backoff=1.0
                )(_call_llm)
            
            # Aplicar timeout si está habilitado
            if timeout and self.enable_timeout:
                _call_llm = async_timeout(timeout=timeout)(_call_llm)
            
            # Ejecutar llamada
            response = await _call_llm()
            
            # Calcular métricas
            elapsed = (datetime.now() - start_time).total_seconds()
            usage = response.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            # Registrar métricas
            if self.metrics_manager:
                self.metrics_manager.record_llm_call(tokens_used, elapsed)
            
            # Actualizar estadísticas
            self.total_calls += 1
            self.total_tokens += tokens_used
            
            # Agregar metadatos a la respuesta
            response["_metadata"] = {
                "tokens_used": tokens_used,
                "response_time": elapsed,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
        
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.total_errors += 1
            
            if self.metrics_manager:
                self.metrics_manager.record_error()
            
            logger.error(f"LLM call failed after {elapsed:.2f}s: {e}", exc_info=True)
            raise LLMError(
                f"LLM call failed: {str(e)}",
                context={"prompt_length": len(prompt), "elapsed": elapsed}
            ) from e
    
    async def generate_with_prompt_builder(
        self,
        prompt_type: str,
        **prompt_kwargs
    ) -> Dict[str, Any]:
        """
        Generar texto usando PromptBuilder para construir el prompt.
        
        Args:
            prompt_type: Tipo de prompt (thinking, react_thought, etc.)
            **prompt_kwargs: Argumentos para el PromptBuilder
            
        Returns:
            Respuesta del LLM
        """
        # Construir prompt usando PromptBuilder
        prompt_method = getattr(PromptBuilder, f"build_{prompt_type}_prompt", None)
        if not prompt_method:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        prompt = prompt_method(**prompt_kwargs)
        
        # Llamar al LLM
        return await self.generate_text(prompt=prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del servicio.
        
        Returns:
            Dict con estadísticas
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "average_tokens_per_call": (
                self.total_tokens / self.total_calls
                if self.total_calls > 0 else 0
            ),
            "error_rate": (
                self.total_errors / self.total_calls
                if self.total_calls > 0 else 0.0
            )
        }
    
    def reset_stats(self) -> None:
        """Resetear estadísticas."""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_errors = 0


class LLMCallTracker:
    """Tracker para llamadas LLM individuales."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.call_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    async def track_call(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Realizar llamada LLM con tracking.
        
        Args:
            prompt: Prompt a enviar
            **kwargs: Argumentos adicionales
            
        Returns:
            Respuesta con tracking
        """
        call_info = {
            "prompt": prompt[:200],  # Truncar para almacenamiento
            "timestamp": datetime.now().isoformat(),
            "kwargs": kwargs
        }
        
        try:
            response = await self.llm_service.generate_text(prompt=prompt, **kwargs)
            call_info["success"] = True
            call_info["tokens_used"] = response.get("_metadata", {}).get("tokens_used", 0)
        except Exception as e:
            call_info["success"] = False
            call_info["error"] = str(e)
            raise
        
        finally:
            # Agregar al historial
            self.call_history.append(call_info)
            if len(self.call_history) > self.max_history:
                self.call_history.pop(0)
        
        return response
    
    def get_recent_calls(self, count: int = 10) -> List[Dict[str, Any]]:
        """Obtener llamadas recientes."""
        return self.call_history[-count:]
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de llamadas."""
        if not self.call_history:
            return {"total": 0, "successful": 0, "failed": 0}
        
        successful = sum(1 for call in self.call_history if call.get("success", False))
        failed = len(self.call_history) - successful
        
        return {
            "total": len(self.call_history),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self.call_history) if self.call_history else 0.0
        }


