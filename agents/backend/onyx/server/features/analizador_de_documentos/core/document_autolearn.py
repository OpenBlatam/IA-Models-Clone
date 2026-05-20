"""
Document Auto-Learning - Sistema de Auto-Aprendizaje
====================================================

Sistema de auto-aprendizaje para mejorar análisis basado en feedback.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class LearningExample:
    """Ejemplo de aprendizaje."""
    example_id: str
    document_content: str
    expected_result: Dict[str, Any]
    actual_result: Dict[str, Any]
    feedback: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningPattern:
    """Patrón de aprendizaje."""
    pattern_id: str
    pattern_type: str
    features: Dict[str, Any]
    confidence: float
    examples_count: int
    learned_at: datetime = field(default_factory=datetime.now)


class AutoLearningSystem:
    """Sistema de auto-aprendizaje."""
    
    def __init__(self, analyzer):
        """Inicializar sistema."""
        self.analyzer = analyzer
        self.examples: List[LearningExample] = []
        self.patterns: List[LearningPattern] = []
        self.max_examples = 10000
        self.learning_enabled = True
    
    def add_learning_example(
        self,
        document_content: str,
        expected_result: Dict[str, Any],
        actual_result: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ) -> LearningExample:
        """
        Agregar ejemplo de aprendizaje.
        
        Args:
            document_content: Contenido del documento
            expected_result: Resultado esperado
            actual_result: Resultado actual
            feedback: Feedback adicional
        
        Returns:
            LearningExample creado
        """
        example = LearningExample(
            example_id=f"example_{len(self.examples) + 1}",
            document_content=document_content,
            expected_result=expected_result,
            actual_result=actual_result,
            feedback=feedback or {}
        )
        
        self.examples.append(example)
        
        # Mantener solo últimos N ejemplos
        if len(self.examples) > self.max_examples:
            self.examples = self.examples[-self.max_examples:]
        
        logger.info(f"Ejemplo de aprendizaje agregado: {example.example_id}")
        
        return example
    
    async def learn_from_examples(self):
        """Aprender de ejemplos."""
        if not self.learning_enabled:
            return
        
        if len(self.examples) < 10:
            logger.warning("Se requieren al menos 10 ejemplos para aprender")
            return
        
        # Analizar patrones comunes
        patterns = self._extract_patterns()
        
        # Actualizar patrones existentes o crear nuevos
        for pattern in patterns:
            existing = next(
                (p for p in self.patterns if p.pattern_id == pattern.pattern_id),
                None
            )
            
            if existing:
                # Actualizar patrón existente
                existing.confidence = (existing.confidence + pattern.confidence) / 2
                existing.examples_count += pattern.examples_count
            else:
                # Agregar nuevo patrón
                self.patterns.append(pattern)
        
        logger.info(f"Patrones aprendidos: {len(self.patterns)}")
    
    def _extract_patterns(self) -> List[LearningPattern]:
        """Extraer patrones de ejemplos."""
        patterns = []
        
        # Agrupar por tipo de error/discrepancia
        error_types = defaultdict(list)
        
        for example in self.examples:
            # Comparar expected vs actual
            discrepancies = self._find_discrepancies(
                example.expected_result,
                example.actual_result
            )
            
            for disc_type, disc_data in discrepancies.items():
                error_types[disc_type].append({
                    "example": example,
                    "data": disc_data
                })
        
        # Crear patrones de errores comunes
        for error_type, examples in error_types.items():
            if len(examples) >= 3:  # Mínimo 3 ejemplos para patrón
                pattern = LearningPattern(
                    pattern_id=f"pattern_{error_type}_{len(patterns)}",
                    pattern_type=error_type,
                    features=self._extract_features(examples),
                    confidence=min(1.0, len(examples) / 10.0),
                    examples_count=len(examples)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_discrepancies(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encontrar discrepancias entre expected y actual."""
        discrepancies = {}
        
        # Comparar scores
        if "quality_score" in expected and "quality_score" in actual:
            diff = abs(expected["quality_score"] - actual["quality_score"])
            if diff > 10:
                discrepancies["quality_score_diff"] = {
                    "expected": expected["quality_score"],
                    "actual": actual["quality_score"],
                    "diff": diff
                }
        
        # Comparar clasificaciones
        if "classification" in expected and "classification" in actual:
            expected_top = max(expected["classification"].items(), key=lambda x: x[1])[0]
            actual_top = max(actual["classification"].items(), key=lambda x: x[1])[0]
            
            if expected_top != actual_top:
                discrepancies["classification_mismatch"] = {
                    "expected": expected_top,
                    "actual": actual_top
                }
        
        return discrepancies
    
    def _extract_features(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extraer features comunes de ejemplos."""
        features = {
            "common_errors": [],
            "average_discrepancy": 0.0,
            "example_count": len(examples)
        }
        
        # Calcular discrepancias promedio
        total_diff = 0.0
        for example_data in examples:
            example = example_data["example"]
            discrepancies = self._find_discrepancies(
                example.expected_result,
                example.actual_result
            )
            
            if discrepancies:
                total_diff += sum(
                    abs(d.get("diff", 0)) for d in discrepancies.values()
                    if isinstance(d, dict) and "diff" in d
                )
        
        if examples:
            features["average_discrepancy"] = total_diff / len(examples)
        
        return features
    
    async def apply_learned_patterns(
        self,
        analysis_result: Any
    ) -> Dict[str, Any]:
        """
        Aplicar patrones aprendidos a resultado.
        
        Args:
            analysis_result: Resultado de análisis
        
        Returns:
            Resultado ajustado
        """
        if not self.patterns:
            return analysis_result
        
        adjusted = {}
        
        # Aplicar ajustes basados en patrones
        for pattern in self.patterns:
            if pattern.confidence > 0.7:  # Solo patrones con alta confianza
                # Ajustar según patrón
                if pattern.pattern_type == "quality_score_diff":
                    # Ajustar score de calidad
                    if hasattr(analysis_result, 'quality_score'):
                        adjustment = pattern.features.get("average_discrepancy", 0) * 0.1
                        adjusted["quality_score_adjustment"] = adjustment
        
        return adjusted
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de aprendizaje."""
        return {
            "total_examples": len(self.examples),
            "total_patterns": len(self.patterns),
            "learning_enabled": self.learning_enabled,
            "patterns_by_type": {
                pattern.pattern_type: len([p for p in self.patterns if p.pattern_type == pattern.pattern_type])
                for pattern in self.patterns
            }
        }


__all__ = [
    "AutoLearningSystem",
    "LearningExample",
    "LearningPattern"
]
















