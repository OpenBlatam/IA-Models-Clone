"""
Model Selector - Selección inteligente de modelos.

Sigue principios de selección de modelos en deep learning.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.logging_config import get_logger
from .model_registry import ModelRegistry, ModelConfig, get_model_registry

logger = get_logger(__name__)


class SelectionStrategy(str, Enum):
    """Estrategias de selección de modelos."""
    BEST_QUALITY = "best_quality"
    COST_EFFICIENT = "cost_efficient"
    FASTEST = "fastest"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class SelectionCriteria:
    """Criterios para selección de modelo."""
    use_case: str
    strategy: SelectionStrategy = SelectionStrategy.BALANCED
    max_cost: Optional[float] = None
    min_quality: Optional[float] = None
    max_latency_ms: Optional[float] = None
    required_capabilities: List[str] = None
    preferred_providers: List[str] = None
    
    def __post_init__(self):
        if self.required_capabilities is None:
            self.required_capabilities = []
        if self.preferred_providers is None:
            self.preferred_providers = []


class ModelSelector:
    """
    Selector inteligente de modelos.
    
    Selecciona el mejor modelo según criterios.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """
        Inicializar selector.
        
        Args:
            registry: Registry de modelos (usa global si None)
        """
        self.registry = registry or get_model_registry()
    
    def select_model(
        self,
        criteria: SelectionCriteria
    ) -> Optional[ModelConfig]:
        """
        Seleccionar modelo según criterios.
        
        Args:
            criteria: Criterios de selección
            
        Returns:
            ModelConfig del modelo seleccionado o None
        """
        # Obtener candidatos
        candidates = self._get_candidates(criteria)
        
        if not candidates:
            logger.warning(f"No se encontraron modelos que cumplan los criterios")
            return None
        
        # Aplicar estrategia de selección
        if criteria.strategy == SelectionStrategy.BEST_QUALITY:
            return self._select_best_quality(candidates)
        elif criteria.strategy == SelectionStrategy.COST_EFFICIENT:
            return self._select_cost_efficient(candidates)
        elif criteria.strategy == SelectionStrategy.FASTEST:
            return self._select_fastest(candidates)
        elif criteria.strategy == SelectionStrategy.BALANCED:
            return self._select_balanced(candidates)
        else:
            return candidates[0]
    
    def _get_candidates(
        self,
        criteria: SelectionCriteria
    ) -> List[ModelConfig]:
        """Obtener candidatos que cumplen criterios básicos."""
        # Obtener modelos recomendados para el caso de uso
        recommended = self.registry.get_recommended_model(criteria.use_case)
        
        if recommended:
            candidates = [recommended]
        else:
            # Buscar por capacidades
            candidates = self.registry.list_models(
                capability=criteria.required_capabilities[0]
                if criteria.required_capabilities
                else None
            )
        
        # Filtrar por criterios
        filtered = []
        for model in candidates:
            # Filtrar por proveedor preferido
            if criteria.preferred_providers:
                if model.provider.value not in criteria.preferred_providers:
                    continue
            
            # Filtrar por capacidades requeridas
            if criteria.required_capabilities:
                if not all(
                    cap in model.capabilities
                    for cap in criteria.required_capabilities
                ):
                    continue
            
            filtered.append(model)
        
        return filtered
    
    def _select_best_quality(
        self,
        candidates: List[ModelConfig]
    ) -> ModelConfig:
        """Seleccionar modelo de mejor calidad."""
        # Ordenar por max_tokens (proxy de calidad)
        return max(candidates, key=lambda m: m.max_tokens)
    
    def _select_cost_efficient(
        self,
        candidates: List[ModelConfig]
    ) -> ModelConfig:
        """Seleccionar modelo más económico."""
        # Ordenar por costo total (input + output)
        return min(
            candidates,
            key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output
        )
    
    def _select_fastest(
        self,
        candidates: List[ModelConfig]
    ) -> ModelConfig:
        """Seleccionar modelo más rápido."""
        # Asumir que modelos más pequeños son más rápidos
        # (esto es una aproximación, en producción usarías métricas reales)
        return min(candidates, key=lambda m: m.max_tokens)
    
    def _select_balanced(
        self,
        candidates: List[ModelConfig]
    ) -> ModelConfig:
        """Seleccionar modelo balanceado."""
        # Score balanceado: calidad vs costo
        def balance_score(model: ModelConfig) -> float:
            # Normalizar calidad (max_tokens) y costo
            max_quality = max(m.max_tokens for m in candidates)
            max_cost = max(
                m.cost_per_1k_input + m.cost_per_1k_output
                for m in candidates
            )
            
            quality_score = model.max_tokens / max_quality if max_quality > 0 else 0
            cost_score = 1.0 - (
                (model.cost_per_1k_input + model.cost_per_1k_output) / max_cost
                if max_cost > 0
                else 0
            )
            
            # Balance: 60% calidad, 40% costo
            return 0.6 * quality_score + 0.4 * cost_score
        
        return max(candidates, key=balance_score)
    
    def select_models_for_parallel(
        self,
        criteria: SelectionCriteria,
        count: int = 3
    ) -> List[ModelConfig]:
        """
        Seleccionar múltiples modelos para ejecución paralela.
        
        Args:
            criteria: Criterios de selección
            count: Número de modelos a seleccionar
            
        Returns:
            Lista de ModelConfig
        """
        candidates = self._get_candidates(criteria)
        
        if not candidates:
            return []
        
        # Seleccionar modelos diversos (diferentes proveedores)
        selected = []
        used_providers = set()
        
        for model in candidates:
            if len(selected) >= count:
                break
            
            # Preferir modelos de diferentes proveedores
            if model.provider.value not in used_providers or len(used_providers) >= len(candidates):
                selected.append(model)
                used_providers.add(model.provider.value)
        
        # Si no tenemos suficientes, agregar más
        while len(selected) < count and len(selected) < len(candidates):
            for model in candidates:
                if model not in selected:
                    selected.append(model)
                    break
        
        return selected[:count]


# Instancia global
_model_selector = ModelSelector()


def get_model_selector() -> ModelSelector:
    """Obtener instancia global del selector."""
    return _model_selector



