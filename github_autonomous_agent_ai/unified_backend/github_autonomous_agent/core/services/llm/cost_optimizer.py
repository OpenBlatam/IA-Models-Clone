"""
Cost Optimizer - Optimización de costos para LLMs.

Sigue principios de optimización de recursos.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from config.logging_config import get_logger
from .model_registry import ModelRegistry, ModelConfig, get_model_registry

logger = get_logger(__name__)


@dataclass
class CostBudget:
    """Presupuesto de costos."""
    daily_limit: float = 10.0
    monthly_limit: float = 300.0
    per_request_limit: float = 0.10
    warning_threshold: float = 0.8  # 80% del límite


@dataclass
class CostStats:
    """Estadísticas de costos."""
    total_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    requests_count: int = 0
    cost_by_model: Dict[str, float] = None
    cost_by_use_case: Dict[str, float] = None
    daily_history: List[Tuple[str, float]] = None
    
    def __post_init__(self):
        if self.cost_by_model is None:
            self.cost_by_model = defaultdict(float)
        if self.cost_by_use_case is None:
            self.cost_by_use_case = defaultdict(float)
        if self.daily_history is None:
            self.daily_history = []


class CostOptimizer:
    """
    Optimizador de costos para LLM Service.
    
    Características:
    - Tracking de costos
    - Presupuestos y límites
    - Recomendaciones de optimización
    - Alertas de presupuesto
    """
    
    def __init__(
        self,
        budget: Optional[CostBudget] = None,
        registry: Optional[ModelRegistry] = None
    ):
        """
        Inicializar optimizador.
        
        Args:
            budget: Presupuesto de costos
            registry: Registry de modelos
        """
        self.budget = budget or CostBudget()
        self.registry = registry or get_model_registry()
        self.stats = CostStats()
        self.last_reset_date = datetime.now().date()
    
    def estimate_cost(
        self,
        model_id: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int
    ) -> float:
        """
        Estimar costo de un request.
        
        Args:
            model_id: ID del modelo
            estimated_input_tokens: Tokens de entrada estimados
            estimated_output_tokens: Tokens de salida estimados
            
        Returns:
            Costo estimado en dólares
        """
        return self.registry.estimate_cost(
            model_id,
            estimated_input_tokens,
            estimated_output_tokens
        )
    
    def record_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        use_case: Optional[str] = None
    ) -> Tuple[float, bool]:
        """
        Registrar costo de un request.
        
        Args:
            model_id: ID del modelo
            input_tokens: Tokens de entrada
            output_tokens: Tokens de salida
            use_case: Caso de uso (opcional)
            
        Returns:
            Tupla (costo, dentro_del_presupuesto)
        """
        cost = self.registry.estimate_cost(model_id, input_tokens, output_tokens)
        
        # Actualizar estadísticas
        self.stats.total_cost += cost
        self.stats.daily_cost += cost
        self.stats.requests_count += 1
        self.stats.cost_by_model[model_id] += cost
        
        if use_case:
            self.stats.cost_by_use_case[use_case] += cost
        
        # Verificar límites
        within_budget = self._check_budget()
        
        # Resetear contadores diarios si es nuevo día
        self._reset_daily_if_needed()
        
        return cost, within_budget
    
    def _check_budget(self) -> bool:
        """Verificar si estamos dentro del presupuesto."""
        # Verificar límite diario
        if self.stats.daily_cost > self.budget.daily_limit:
            logger.warning(
                f"Límite diario excedido: ${self.stats.daily_cost:.2f} / ${self.budget.daily_limit:.2f}"
            )
            return False
        
        # Verificar límite mensual
        if self.stats.monthly_cost > self.budget.monthly_limit:
            logger.warning(
                f"Límite mensual excedido: ${self.stats.monthly_cost:.2f} / ${self.budget.monthly_limit:.2f}"
            )
            return False
        
        # Verificar umbral de advertencia
        daily_ratio = self.stats.daily_cost / self.budget.daily_limit
        if daily_ratio >= self.budget.warning_threshold:
            logger.warning(
                f"Acercándose al límite diario: {daily_ratio*100:.1f}% usado"
            )
        
        return True
    
    def _reset_daily_if_needed(self) -> None:
        """Resetear contadores diarios si es nuevo día."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            # Guardar histórico
            self.stats.daily_history.append((
                self.last_reset_date.isoformat(),
                self.stats.daily_cost
            ))
            
            # Resetear
            self.stats.daily_cost = 0.0
            self.last_reset_date = today
            
            # Actualizar mensual (simplificado: suma últimos 30 días)
            if len(self.stats.daily_history) > 30:
                self.stats.daily_history = self.stats.daily_history[-30:]
            
            self.stats.monthly_cost = sum(cost for _, cost in self.stats.daily_history)
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones de optimización.
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar costos por modelo
        if self.stats.cost_by_model:
            most_expensive = max(
                self.stats.cost_by_model.items(),
                key=lambda x: x[1]
            )
            
            model_config = self.registry.get(most_expensive[0])
            if model_config:
                # Buscar alternativa más económica
                alternatives = self.registry.list_models()
                cheaper = [
                    m for m in alternatives
                    if (m.cost_per_1k_input + m.cost_per_1k_output) <
                       (model_config.cost_per_1k_input + model_config.cost_per_1k_output)
                    and m.model_id != most_expensive[0]
                ]
                
                if cheaper:
                    cheapest = min(
                        cheaper,
                        key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output
                    )
                    savings = (
                        (model_config.cost_per_1k_input + model_config.cost_per_1k_output) -
                        (cheapest.cost_per_1k_input + cheapest.cost_per_1k_output)
                    ) * (most_expensive[1] / (model_config.cost_per_1k_input + model_config.cost_per_1k_output))
                    
                    recommendations.append({
                        "type": "model_replacement",
                        "current_model": most_expensive[0],
                        "recommended_model": cheapest.model_id,
                        "estimated_savings": savings,
                        "message": f"Considera usar {cheapest.model_id} en lugar de {most_expensive[0]} para ahorrar ~${savings:.2f}"
                    })
        
        # Verificar uso de caché
        cache_hit_rate = getattr(self, '_cache_hit_rate', 0.0)
        if cache_hit_rate < 0.3:
            recommendations.append({
                "type": "cache_optimization",
                "message": f"Cache hit rate bajo ({cache_hit_rate*100:.1f}%). Considera aumentar TTL o tamaño de caché."
            })
        
        # Verificar distribución de costos
        if len(self.stats.cost_by_use_case) > 1:
            most_expensive_use_case = max(
                self.stats.cost_by_use_case.items(),
                key=lambda x: x[1]
            )
            recommendations.append({
                "type": "use_case_optimization",
                "use_case": most_expensive_use_case[0],
                "cost": most_expensive_use_case[1],
                "message": f"El caso de uso '{most_expensive_use_case[0]}' consume ${most_expensive_use_case[1]:.2f}. Considera optimizar."
            })
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de costos."""
        return {
            "total_cost": self.stats.total_cost,
            "daily_cost": self.stats.daily_cost,
            "monthly_cost": self.stats.monthly_cost,
            "requests_count": self.stats.requests_count,
            "average_cost_per_request": (
                self.stats.total_cost / self.stats.requests_count
                if self.stats.requests_count > 0
                else 0.0
            ),
            "budget": {
                "daily_limit": self.budget.daily_limit,
                "monthly_limit": self.budget.monthly_limit,
                "daily_usage_percent": (
                    self.stats.daily_cost / self.budget.daily_limit * 100
                    if self.budget.daily_limit > 0
                    else 0.0
                ),
                "monthly_usage_percent": (
                    self.stats.monthly_cost / self.budget.monthly_limit * 100
                    if self.budget.monthly_limit > 0
                    else 0.0
                )
            },
            "cost_by_model": dict(self.stats.cost_by_model),
            "cost_by_use_case": dict(self.stats.cost_by_use_case),
            "recommendations": self.get_recommendations()
        }
    
    def reset_stats(self) -> None:
        """Resetear estadísticas."""
        self.stats = CostStats()
        self.last_reset_date = datetime.now().date()
        logger.info("Estadísticas de costos reseteadas")


# Instancia global
_cost_optimizer = CostOptimizer()


def get_cost_optimizer() -> CostOptimizer:
    """Obtener instancia global del optimizador."""
    return _cost_optimizer



