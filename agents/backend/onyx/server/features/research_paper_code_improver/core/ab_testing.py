"""
A/B Testing Framework - Framework para A/B testing de modelos
===============================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy no disponible, algunas funcionalidades estarán limitadas")

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuración de A/B test"""
    model_a_name: str
    model_b_name: str
    traffic_split: float = 0.5  # 0.5 = 50/50 split
    metric: str = "accuracy"
    min_samples: int = 100
    confidence_level: float = 0.95


@dataclass
class ABTestResult:
    """Resultado de A/B test"""
    model_a_metric: float
    model_b_metric: float
    difference: float
    p_value: float
    is_significant: bool
    winner: Optional[str] = None
    confidence_interval: tuple = (0.0, 0.0)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "model_a_metric": self.model_a_metric,
            "model_b_metric": self.model_b_metric,
            "difference": self.difference,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "winner": self.winner,
            "confidence_interval": self.confidence_interval,
            "timestamp": self.timestamp.isoformat()
        }


class ABTestingFramework:
    """Framework para A/B testing"""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.model_a_results: List[float] = []
        self.model_b_results: List[float] = []
        self.test_results: List[ABTestResult] = []
    
    def assign_model(self, user_id: str) -> str:
        """Asigna modelo A o B basado en user_id"""
        hash_value = hash(user_id) % 100
        threshold = self.config.traffic_split * 100
        
        if hash_value < threshold:
            return self.config.model_a_name
        else:
            return self.config.model_b_name
    
    def record_result(
        self,
        model_name: str,
        metric_value: float
    ):
        """Registra resultado de un modelo"""
        if model_name == self.config.model_a_name:
            self.model_a_results.append(metric_value)
        elif model_name == self.config.model_b_name:
            self.model_b_results.append(metric_value)
    
    def run_statistical_test(self) -> ABTestResult:
        """Ejecuta test estadístico"""
        if (len(self.model_a_results) < self.config.min_samples or
            len(self.model_b_results) < self.config.min_samples):
            logger.warning("No hay suficientes muestras para test estadístico")
            return None
        
        # Calcular métricas
        metric_a = np.mean(self.model_a_results)
        metric_b = np.mean(self.model_b_results)
        difference = metric_b - metric_a
        
        # Test t de Student
        if SCIPY_AVAILABLE:
            t_stat, p_value = stats.ttest_ind(
                self.model_a_results,
                self.model_b_results
            )
            
            # Intervalo de confianza
            se_a = np.std(self.model_a_results) / np.sqrt(len(self.model_a_results))
            se_b = np.std(self.model_b_results) / np.sqrt(len(self.model_b_results))
            se_diff = np.sqrt(se_a**2 + se_b**2)
            
            alpha = 1 - self.config.confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, len(self.model_a_results) + len(self.model_b_results) - 2)
            margin = t_critical * se_diff
        else:
            # Fallback sin scipy
            p_value = 0.5  # Placeholder
            margin = np.std(self.model_a_results + self.model_b_results) * 1.96
        
        ci_lower = difference - margin
        ci_upper = difference + margin
        
        # Determinar si es significativo
        is_significant = p_value < (1 - self.config.confidence_level)
        
        # Determinar ganador
        winner = None
        if is_significant:
            winner = self.config.model_b_name if difference > 0 else self.config.model_a_name
        
        result = ABTestResult(
            model_a_metric=metric_a,
            model_b_metric=metric_b,
            difference=difference,
            p_value=p_value,
            is_significant=is_significant,
            winner=winner,
            confidence_interval=(ci_lower, ci_upper)
        )
        
        self.test_results.append(result)
        return result
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del test"""
        if not self.test_results:
            return {}
        
        latest_result = self.test_results[-1]
        
        return {
            "total_samples_a": len(self.model_a_results),
            "total_samples_b": len(self.model_b_results),
            "latest_result": latest_result.to_dict(),
            "all_results": [r.to_dict() for r in self.test_results]
        }
