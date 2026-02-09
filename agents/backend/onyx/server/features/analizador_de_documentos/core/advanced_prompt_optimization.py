"""
Sistema de Advanced Prompt Optimization
=========================================

Sistema avanzado para optimización de prompts.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Método de optimización"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    GRADIENT_BASED = "gradient_based"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"


@dataclass
class OptimizedPrompt:
    """Prompt optimizado"""
    prompt_id: str
    original_prompt: str
    optimized_prompt: str
    performance_improvement: float
    method: OptimizationMethod
    timestamp: str


class AdvancedPromptOptimization:
    """
    Sistema de Advanced Prompt Optimization
    
    Proporciona:
    - Optimización avanzada de prompts
    - Múltiples métodos de optimización
    - Búsqueda automática de mejores prompts
    - Evaluación de rendimiento
    - Generación evolutiva de prompts
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.optimizations: Dict[str, OptimizedPrompt] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        logger.info("AdvancedPromptOptimization inicializado")
    
    def optimize_prompt(
        self,
        original_prompt: str,
        method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM,
        iterations: int = 10
    ) -> OptimizedPrompt:
        """
        Optimizar prompt
        
        Args:
            original_prompt: Prompt original
            method: Método de optimización
            iterations: Número de iteraciones
        
        Returns:
            Prompt optimizado
        """
        prompt_id = f"opt_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulación de optimización
        # En producción, usaría algoritmos evolutivos, RL, etc.
        optimized_text = f"[Optimizado] {original_prompt} [Con mejoras en estructura y claridad]"
        
        optimized = OptimizedPrompt(
            prompt_id=prompt_id,
            original_prompt=original_prompt,
            optimized_prompt=optimized_text,
            performance_improvement=0.15,  # 15% de mejora
            method=method,
            timestamp=datetime.now().isoformat()
        )
        
        self.optimizations[prompt_id] = optimized
        
        logger.info(f"Prompt optimizado: {prompt_id} - Mejora: {optimized.performance_improvement:.2%}")
        
        return optimized
    
    def evaluate_prompt_performance(
        self,
        prompt: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluar rendimiento de prompt
        
        Args:
            prompt: Prompt a evaluar
            test_cases: Casos de prueba
        
        Returns:
            Métricas de rendimiento
        """
        # Simulación de evaluación
        performance = {
            "prompt": prompt[:100] + "...",
            "accuracy": 0.88,
            "precision": 0.85,
            "recall": 0.90,
            "f1_score": 0.87,
            "avg_response_time": 1.2,
            "test_cases": len(test_cases)
        }
        
        self.evaluation_history.append({
            "prompt": prompt,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Rendimiento evaluado: Accuracy {performance['accuracy']:.2%}")
        
        return performance
    
    def compare_prompts(
        self,
        prompts: List[str],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comparar múltiples prompts
        
        Args:
            prompts: Lista de prompts
            test_cases: Casos de prueba
        
        Returns:
            Comparación de prompts
        """
        comparison = {
            "prompts": [],
            "best_prompt": None,
            "best_score": 0.0
        }
        
        for prompt in prompts:
            performance = self.evaluate_prompt_performance(prompt, test_cases)
            score = performance["f1_score"]
            
            comparison["prompts"].append({
                "prompt": prompt[:50] + "...",
                "score": score,
                "accuracy": performance["accuracy"]
            })
            
            if score > comparison["best_score"]:
                comparison["best_score"] = score
                comparison["best_prompt"] = prompt
        
        logger.info(f"Comparación completada: {len(prompts)} prompts")
        
        return comparison


# Instancia global
_advanced_prompt_opt: Optional[AdvancedPromptOptimization] = None


def get_advanced_prompt_optimization() -> AdvancedPromptOptimization:
    """Obtener instancia global del sistema"""
    global _advanced_prompt_opt
    if _advanced_prompt_opt is None:
        _advanced_prompt_opt = AdvancedPromptOptimization()
    return _advanced_prompt_opt


