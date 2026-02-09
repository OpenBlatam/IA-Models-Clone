"""
Document Auto Optimizer - Optimización Automática
=================================================

Sistema de optimización automática de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuggestion:
    """Sugerencia de optimización."""
    suggestion_type: str
    description: str
    original_text: str
    optimized_text: str
    improvement_score: float
    priority: str  # 'high', 'medium', 'low'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Resultado de optimización."""
    document_id: str
    original_score: float
    optimized_score: float
    improvement: float
    suggestions: List[OptimizationSuggestion]
    optimized_content: str
    applied_optimizations: List[str] = field(default_factory=list)


class AutoOptimizer:
    """Optimizador automático."""
    
    def __init__(self, analyzer):
        """Inicializar optimizador."""
        self.analyzer = analyzer
    
    async def optimize_document(
        self,
        document_id: str,
        content: str,
        optimization_goals: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Optimizar documento automáticamente.
        
        Args:
            document_id: ID del documento
            content: Contenido a optimizar
            optimization_goals: Objetivos de optimización (clarity, brevity, engagement, etc.)
        
        Returns:
            OptimizationResult con resultados
        """
        optimization_goals = optimization_goals or ["clarity", "brevity"]
        
        suggestions = []
        optimized_content = content
        
        # Calcular score original
        original_score = await self._calculate_document_score(content)
        
        # Generar sugerencias de optimización
        if "clarity" in optimization_goals:
            clarity_suggestions = await self._optimize_clarity(content)
            suggestions.extend(clarity_suggestions)
        
        if "brevity" in optimization_goals:
            brevity_suggestions = await self._optimize_brevity(content)
            suggestions.extend(brevity_suggestions)
        
        if "engagement" in optimization_goals:
            engagement_suggestions = await self._optimize_engagement(content)
            suggestions.extend(engagement_suggestions)
        
        # Aplicar optimizaciones
        applied_optimizations = []
        for suggestion in suggestions:
            if suggestion.priority == "high":
                optimized_content = optimized_content.replace(
                    suggestion.original_text,
                    suggestion.optimized_text
                )
                applied_optimizations.append(suggestion.suggestion_type)
        
        # Calcular score optimizado
        optimized_score = await self._calculate_document_score(optimized_content)
        improvement = optimized_score - original_score
        
        return OptimizationResult(
            document_id=document_id,
            original_score=original_score,
            optimized_score=optimized_score,
            improvement=improvement,
            suggestions=suggestions,
            optimized_content=optimized_content,
            applied_optimizations=applied_optimizations
        )
    
    async def _calculate_document_score(self, content: str) -> float:
        """Calcular score del documento."""
        score = 0.0
        
        # Longitud apropiada
        word_count = len(content.split())
        if 200 <= word_count <= 2000:
            score += 0.3
        elif word_count < 200:
            score += 0.1
        
        # Variedad de palabras
        words = content.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        score += unique_ratio * 0.2
        
        # Estructura
        paragraphs = content.split('\n\n')
        if 3 <= len(paragraphs) <= 10:
            score += 0.2
        
        # Complejidad de oraciones
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 10 <= avg_sentence_length <= 25:
            score += 0.2
        
        # Puntuación
        has_punctuation = any(c in content for c in '.,!?;:')
        if has_punctuation:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _optimize_clarity(self, content: str) -> List[OptimizationSuggestion]:
        """Optimizar claridad."""
        suggestions = []
        
        # Detectar frases confusas comunes
        confusing_patterns = [
            ("muy muy", "extremadamente"),
            ("bastante bastante", "considerablemente"),
            ("en el caso de que", "si"),
            ("debido al hecho de que", "porque"),
            ("a pesar de que", "aunque")
        ]
        
        for original, replacement in confusing_patterns:
            if original in content.lower():
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="clarity",
                    description=f"Simplificar frase: '{original}'",
                    original_text=original,
                    optimized_text=replacement,
                    improvement_score=0.3,
                    priority="medium"
                ))
        
        return suggestions
    
    async def _optimize_brevity(self, content: str) -> List[OptimizationSuggestion]:
        """Optimizar brevedad."""
        suggestions = []
        
        # Detectar redundancias
        redundant_patterns = [
            ("completamente lleno", "lleno"),
            ("finalizar completamente", "finalizar"),
            ("comenzar a empezar", "comenzar"),
            ("repetir de nuevo", "repetir"),
            ("plan futuro", "plan")
        ]
        
        for original, replacement in redundant_patterns:
            if original in content.lower():
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="brevity",
                    description=f"Eliminar redundancia: '{original}'",
                    original_text=original,
                    optimized_text=replacement,
                    improvement_score=0.2,
                    priority="low"
                ))
        
        return suggestions
    
    async def _optimize_engagement(self, content: str) -> List[OptimizationSuggestion]:
        """Optimizar engagement."""
        suggestions = []
        
        # Detectar lenguaje pasivo
        passive_patterns = [
            ("fue realizado", "realizamos"),
            ("es considerado", "consideramos"),
            ("fue implementado", "implementamos")
        ]
        
        for original, replacement in passive_patterns:
            if original in content.lower():
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="engagement",
                    description=f"Usar voz activa: '{original}' -> '{replacement}'",
                    original_text=original,
                    optimized_text=replacement,
                    improvement_score=0.25,
                    priority="medium"
                ))
        
        return suggestions


__all__ = [
    "AutoOptimizer",
    "OptimizationResult",
    "OptimizationSuggestion"
]


