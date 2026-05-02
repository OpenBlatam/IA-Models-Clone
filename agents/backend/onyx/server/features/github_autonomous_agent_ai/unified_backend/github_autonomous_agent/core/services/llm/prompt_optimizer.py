"""
Prompt Optimizer - Optimización automática de prompts.

Analiza y optimiza prompts para mejorar:
- Claridad
- Eficiencia (reducir tokens)
- Resultados (mejor calidad de respuesta)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from config.logging_config import get_logger

logger = get_logger(__name__)


class OptimizationGoal(str, Enum):
    """Objetivos de optimización."""
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    COST = "cost"
    BALANCED = "balanced"


@dataclass
class PromptAnalysis:
    """Análisis de un prompt."""
    original_prompt: str
    optimized_prompt: str
    improvements: List[str]
    estimated_token_reduction: int
    clarity_score: float
    efficiency_score: float
    optimization_goal: OptimizationGoal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "improvements": self.improvements,
            "estimated_token_reduction": self.estimated_token_reduction,
            "clarity_score": self.clarity_score,
            "efficiency_score": self.efficiency_score,
            "optimization_goal": self.optimization_goal.value
        }


class PromptOptimizer:
    """
    Optimizador de prompts.
    
    Características:
    - Análisis de prompts
    - Sugerencias de mejora
    - Optimización automática
    - Reducción de tokens
    - Mejora de claridad
    """
    
    def __init__(self):
        """Inicializar optimizador."""
        # Patrones comunes de optimización
        self.optimization_patterns = {
            "remove_redundancy": [
                (r"\s+", " "),  # Múltiples espacios
                (r"\.{2,}", "."),  # Múltiples puntos
                (r"\?\?+", "?"),  # Múltiples signos de interrogación
            ],
            "improve_clarity": [
                # Reemplazos comunes
                ("por favor", ""),
                ("si es posible", ""),
                ("si puedes", ""),
                ("te agradecería", ""),
            ],
            "reduce_length": [
                # Frases largas a cortas
                ("en el caso de que", "si"),
                ("a pesar de que", "aunque"),
                ("debido a que", "porque"),
                ("con el objetivo de", "para"),
            ]
        }
    
    def analyze_prompt(
        self,
        prompt: str,
        goal: OptimizationGoal = OptimizationGoal.BALANCED
    ) -> PromptAnalysis:
        """
        Analizar un prompt y sugerir optimizaciones.
        
        Args:
            prompt: Prompt a analizar
            goal: Objetivo de optimización
            
        Returns:
            Análisis con sugerencias
        """
        improvements = []
        optimized = prompt
        token_reduction = 0
        
        # Análisis básico
        original_length = len(prompt)
        word_count = len(prompt.split())
        
        # Detectar problemas comunes
        if "por favor" in prompt.lower():
            improvements.append("Eliminar 'por favor' - no es necesario para LLMs")
        
        if prompt.count("?") > 3:
            improvements.append("Demasiadas preguntas - considerar dividir en múltiples prompts")
        
        if len(prompt) > 2000:
            improvements.append("Prompt muy largo - considerar acortar para mejor contexto")
        
        # Optimización según objetivo
        if goal == OptimizationGoal.EFFICIENCY or goal == OptimizationGoal.BALANCED:
            optimized, reduction = self._optimize_for_efficiency(optimized)
            token_reduction += reduction
            if reduction > 0:
                improvements.append(f"Reducción de tokens: ~{reduction}")
        
        if goal == OptimizationGoal.CLARITY or goal == OptimizationGoal.BALANCED:
            optimized = self._optimize_for_clarity(optimized)
            improvements.append("Mejora de claridad aplicada")
        
        # Calcular scores
        clarity_score = self._calculate_clarity_score(optimized)
        efficiency_score = self._calculate_efficiency_score(optimized, original_length)
        
        return PromptAnalysis(
            original_prompt=prompt,
            optimized_prompt=optimized,
            improvements=improvements,
            estimated_token_reduction=token_reduction,
            clarity_score=clarity_score,
            efficiency_score=efficiency_score,
            optimization_goal=goal
        )
    
    def optimize_prompt(
        self,
        prompt: str,
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
        max_iterations: int = 3
    ) -> str:
        """
        Optimizar un prompt automáticamente.
        
        Args:
            prompt: Prompt a optimizar
            goal: Objetivo de optimización
            max_iterations: Máximo de iteraciones
            
        Returns:
            Prompt optimizado
        """
        analysis = self.analyze_prompt(prompt, goal)
        return analysis.optimized_prompt
    
    def _optimize_for_efficiency(self, prompt: str) -> Tuple[str, int]:
        """Optimizar para eficiencia (reducir tokens)."""
        import re
        
        original_length = len(prompt)
        optimized = prompt
        
        # Aplicar patrones de reducción
        for pattern, replacement in self.optimization_patterns["remove_redundancy"]:
            optimized = re.sub(pattern, replacement, optimized)
        
        for old_phrase, new_phrase in self.optimization_patterns["reduce_length"]:
            optimized = optimized.replace(old_phrase, new_phrase)
        
        # Estimar reducción de tokens (aproximado: 1 token ≈ 4 caracteres)
        reduction = (original_length - len(optimized)) // 4
        
        return optimized.strip(), max(0, reduction)
    
    def _optimize_for_clarity(self, prompt: str) -> str:
        """Optimizar para claridad."""
        optimized = prompt
        
        # Aplicar mejoras de claridad
        for old_phrase, new_phrase in self.optimization_patterns["improve_clarity"]:
            optimized = optimized.replace(old_phrase, new_phrase)
        
        # Asegurar que termina con punto si es una instrucción
        if optimized and not optimized[-1] in ".!?":
            if any(word in optimized.lower() for word in ["haz", "crea", "genera", "analiza"]):
                optimized += "."
        
        return optimized.strip()
    
    def _calculate_clarity_score(self, prompt: str) -> float:
        """
        Calcular score de claridad (0-1).
        
        Factores:
        - Longitud apropiada
        - Estructura clara
        - Vocabulario específico
        """
        score = 1.0
        
        # Penalizar prompts muy cortos o muy largos
        if len(prompt) < 10:
            score -= 0.3
        elif len(prompt) > 5000:
            score -= 0.2
        
        # Bonificar estructura clara
        if any(marker in prompt for marker in ["1.", "2.", "-", "*"]):
            score += 0.1
        
        # Bonificar vocabulario específico
        specific_words = ["código", "función", "clase", "archivo", "error", "test"]
        if any(word in prompt.lower() for word in specific_words):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_efficiency_score(self, prompt: str, original_length: int) -> float:
        """
        Calcular score de eficiencia (0-1).
        
        Basado en reducción de longitud manteniendo significado.
        """
        if original_length == 0:
            return 0.0
        
        reduction_ratio = (original_length - len(prompt)) / original_length
        return min(1.0, max(0.0, reduction_ratio))
    
    def suggest_improvements(self, prompt: str) -> List[str]:
        """
        Sugerir mejoras para un prompt sin optimizarlo.
        
        Args:
            prompt: Prompt a analizar
            
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Verificar longitud
        if len(prompt) < 20:
            suggestions.append("Prompt muy corto - agregar más contexto")
        elif len(prompt) > 3000:
            suggestions.append("Prompt muy largo - considerar dividir o resumir")
        
        # Verificar estructura
        if not any(marker in prompt for marker in [".", "!", "?", "\n"]):
            suggestions.append("Agregar puntuación o estructura para mejor legibilidad")
        
        # Verificar especificidad
        vague_words = ["algo", "algunos", "varios", "muchos", "pocos"]
        if any(word in prompt.lower() for word in vague_words):
            suggestions.append("Usar términos más específicos en lugar de palabras vagas")
        
        # Verificar instrucciones claras
        action_words = ["haz", "crea", "genera", "analiza", "explica", "mejora"]
        if not any(word in prompt.lower() for word in action_words):
            suggestions.append("Incluir verbos de acción claros (haz, crea, analiza, etc.)")
        
        return suggestions


def get_prompt_optimizer() -> PromptOptimizer:
    """Factory function para obtener instancia singleton del optimizador."""
    if not hasattr(get_prompt_optimizer, "_instance"):
        get_prompt_optimizer._instance = PromptOptimizer()
    return get_prompt_optimizer._instance



