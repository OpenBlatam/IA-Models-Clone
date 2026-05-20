"""
Model Fallback System - Sistema de fallback automático de modelos.

Si un modelo falla o no está disponible, automáticamente intenta
con modelos alternativos según una estrategia configurada.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from config.logging_config import get_logger

logger = get_logger(__name__)


class FallbackStrategy(str, Enum):
    """Estrategias de fallback."""
    COST_OPTIMIZED = "cost_optimized"  # Modelos más baratos primero
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Modelos más rápidos primero
    QUALITY_OPTIMIZED = "quality_optimized"  # Modelos de mejor calidad primero
    SIMILAR_CAPABILITIES = "similar_capabilities"  # Modelos con capacidades similares
    CUSTOM = "custom"  # Lista personalizada


@dataclass
class FallbackConfig:
    """Configuración de fallback."""
    primary_model: str
    fallback_models: List[str]
    strategy: FallbackStrategy
    max_fallbacks: int = 3
    retry_on_error: bool = True
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "primary_model": self.primary_model,
            "fallback_models": self.fallback_models,
            "strategy": self.strategy.value,
            "max_fallbacks": self.max_fallbacks,
            "retry_on_error": self.retry_on_error,
            "retry_on_timeout": self.retry_on_timeout,
            "retry_on_rate_limit": self.retry_on_rate_limit
        }


@dataclass
class FallbackResult:
    """Resultado de un fallback."""
    model_used: str
    fallback_count: int
    models_tried: List[str]
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "model_used": self.model_used,
            "fallback_count": self.fallback_count,
            "models_tried": self.models_tried,
            "success": self.success,
            "error": self.error
        }


class ModelFallbackSystem:
    """
    Sistema de fallback automático de modelos.
    
    Características:
    - Fallback automático en caso de error
    - Múltiples estrategias de fallback
    - Configuración por modelo
    - Tracking de fallbacks
    """
    
    def __init__(self):
        """Inicializar sistema de fallback."""
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        self.fallback_history: List[FallbackResult] = []
        
        # Modelos por categoría (para estrategias automáticas)
        self.model_categories = {
            "high_quality": [
                "openai/gpt-4",
                "openai/gpt-4-turbo",
                "anthropic/claude-3-opus",
                "anthropic/claude-3.5-sonnet"
            ],
            "balanced": [
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro-1.5",
                "meta-llama/llama-3.1-70b-instruct"
            ],
            "cost_effective": [
                "openai/gpt-4o-mini",
                "google/gemini-pro",
                "meta-llama/llama-3.1-8b-instruct",
                "mistralai/mistral-7b-instruct"
            ],
            "fast": [
                "openai/gpt-4o-mini",
                "google/gemini-pro",
                "meta-llama/llama-3.1-8b-instruct"
            ]
        }
    
    def configure_fallback(
        self,
        primary_model: str,
        fallback_models: Optional[List[str]] = None,
        strategy: FallbackStrategy = FallbackStrategy.BALANCED,
        max_fallbacks: int = 3,
        retry_on_error: bool = True,
        retry_on_timeout: bool = True,
        retry_on_rate_limit: bool = True
    ) -> None:
        """
        Configurar fallback para un modelo.
        
        Args:
            primary_model: Modelo principal
            fallback_models: Lista de modelos de fallback (opcional, se genera si no se proporciona)
            strategy: Estrategia de fallback
            max_fallbacks: Máximo de fallbacks permitidos
            retry_on_error: Si reintentar en errores
            retry_on_timeout: Si reintentar en timeouts
            retry_on_rate_limit: Si reintentar en rate limits
        """
        if fallback_models is None:
            fallback_models = self._generate_fallback_models(primary_model, strategy)
        
        config = FallbackConfig(
            primary_model=primary_model,
            fallback_models=fallback_models,
            strategy=strategy,
            max_fallbacks=max_fallbacks,
            retry_on_error=retry_on_error,
            retry_on_timeout=retry_on_timeout,
            retry_on_rate_limit=retry_on_rate_limit
        )
        
        self.fallback_configs[primary_model] = config
        logger.info(f"Fallback configurado para {primary_model}: {len(fallback_models)} modelos de respaldo")
    
    def get_fallback_models(
        self,
        primary_model: str,
        error_type: Optional[str] = None
    ) -> List[str]:
        """
        Obtener modelos de fallback para un modelo.
        
        Args:
            primary_model: Modelo principal
            error_type: Tipo de error (opcional, para filtrado)
            
        Returns:
            Lista de modelos de fallback
        """
        config = self.fallback_configs.get(primary_model)
        if not config:
            # Generar configuración automática
            self.configure_fallback(primary_model)
            config = self.fallback_configs[primary_model]
        
        # Filtrar según tipo de error
        if error_type == "rate_limit" and not config.retry_on_rate_limit:
            return []
        if error_type == "timeout" and not config.retry_on_timeout:
            return []
        if error_type == "error" and not config.retry_on_error:
            return []
        
        return config.fallback_models[:config.max_fallbacks]
    
    def should_fallback(
        self,
        primary_model: str,
        error: Optional[str] = None,
        error_type: Optional[str] = None
    ) -> bool:
        """
        Determinar si se debe hacer fallback.
        
        Args:
            primary_model: Modelo principal
            error: Mensaje de error (opcional)
            error_type: Tipo de error (opcional)
            
        Returns:
            True si se debe hacer fallback
        """
        config = self.fallback_configs.get(primary_model)
        if not config:
            return True  # Por defecto, permitir fallback
        
        if error_type == "rate_limit":
            return config.retry_on_rate_limit
        if error_type == "timeout":
            return config.retry_on_timeout
        if error_type == "error":
            return config.retry_on_error
        
        return True
    
    def record_fallback(
        self,
        primary_model: str,
        model_used: str,
        fallback_count: int,
        models_tried: List[str],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Registrar resultado de un fallback.
        
        Args:
            primary_model: Modelo principal
            model_used: Modelo finalmente usado
            fallback_count: Número de fallbacks
            models_tried: Lista de modelos intentados
            success: Si fue exitoso
            error: Error si hubo (opcional)
        """
        result = FallbackResult(
            model_used=model_used,
            fallback_count=fallback_count,
            models_tried=models_tried,
            success=success,
            error=error
        )
        
        self.fallback_history.append(result)
        
        # Mantener solo últimos 1000 registros
        if len(self.fallback_history) > 1000:
            self.fallback_history = self.fallback_history[-1000:]
        
        logger.info(
            f"Fallback registrado: {primary_model} -> {model_used} "
            f"(fallbacks: {fallback_count}, éxito: {success})"
        )
    
    def get_fallback_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de fallbacks.
        
        Args:
            model: Modelo específico (opcional, todas si no se proporciona)
            
        Returns:
            Estadísticas de fallbacks
        """
        if model:
            results = [r for r in self.fallback_history if model in r.models_tried]
        else:
            results = self.fallback_history
        
        if not results:
            return {
                "total_fallbacks": 0,
                "successful_fallbacks": 0,
                "failed_fallbacks": 0,
                "avg_fallback_count": 0.0
            }
        
        successful = sum(1 for r in results if r.success)
        avg_fallback = sum(r.fallback_count for r in results) / len(results)
        
        return {
            "total_fallbacks": len(results),
            "successful_fallbacks": successful,
            "failed_fallbacks": len(results) - successful,
            "success_rate": successful / len(results) if results else 0.0,
            "avg_fallback_count": avg_fallback,
            "most_used_fallback": self._get_most_used_fallback(results)
        }
    
    def _generate_fallback_models(
        self,
        primary_model: str,
        strategy: FallbackStrategy
    ) -> List[str]:
        """
        Generar lista de modelos de fallback según estrategia.
        
        Args:
            primary_model: Modelo principal
            strategy: Estrategia de fallback
            
        Returns:
            Lista de modelos de fallback
        """
        if strategy == FallbackStrategy.COST_OPTIMIZED:
            return self.model_categories["cost_effective"].copy()
        elif strategy == FallbackStrategy.PERFORMANCE_OPTIMIZED:
            return self.model_categories["fast"].copy()
        elif strategy == FallbackStrategy.QUALITY_OPTIMIZED:
            return self.model_categories["high_quality"].copy()
        elif strategy == FallbackStrategy.SIMILAR_CAPABILITIES:
            # Intentar encontrar modelos similares
            for category, models in self.model_categories.items():
                if primary_model in models:
                    # Retornar otros modelos de la misma categoría
                    return [m for m in models if m != primary_model]
            # Fallback a balanced
            return self.model_categories["balanced"].copy()
        else:
            # BALANCED o default
            return self.model_categories["balanced"].copy()
    
    def _get_most_used_fallback(self, results: List[FallbackResult]) -> Optional[str]:
        """Obtener modelo de fallback más usado."""
        if not results:
            return None
        
        from collections import Counter
        fallback_models = [
            r.model_used for r in results
            if r.fallback_count > 0
        ]
        
        if not fallback_models:
            return None
        
        counter = Counter(fallback_models)
        return counter.most_common(1)[0][0]


def get_model_fallback_system() -> ModelFallbackSystem:
    """Factory function para obtener instancia singleton del sistema."""
    if not hasattr(get_model_fallback_system, "_instance"):
        get_model_fallback_system._instance = ModelFallbackSystem()
    return get_model_fallback_system._instance



