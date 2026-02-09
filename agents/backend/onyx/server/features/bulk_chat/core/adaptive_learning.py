"""
Adaptive Learning - Sistema de Aprendizaje Adaptativo
=====================================================

Sistema que aprende y se adapta automáticamente basado en datos históricos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """Patrón aprendido."""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    outcome: Any
    confidence: float
    learned_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0


class AdaptiveLearningSystem:
    """Sistema de aprendizaje adaptativo."""
    
    def __init__(self, memory_window_hours: int = 24):
        self.patterns: Dict[str, LearningPattern] = {}
        self.observations: deque = deque(maxlen=10000)
        self.memory_window_hours = memory_window_hours
        self._lock = asyncio.Lock()
    
    async def observe(
        self,
        context: Dict[str, Any],
        action: str,
        outcome: Dict[str, Any],
    ):
        """
        Observar resultado de una acción.
        
        Args:
            context: Contexto de la acción
            action: Acción realizada
            outcome: Resultado de la acción
        """
        observation = {
            "context": context,
            "action": action,
            "outcome": outcome,
            "timestamp": datetime.now(),
        }
        
        async with self._lock:
            self.observations.append(observation)
        
        # Aprender patrones
        await self._learn_patterns()
    
    async def _learn_patterns(self):
        """Aprender patrones de las observaciones."""
        if len(self.observations) < 10:
            return
        
        # Agrupar observaciones por contexto similar
        context_groups: Dict[str, List[Dict]] = defaultdict(list)
        
        for obs in self.observations:
            # Crear key del contexto (simplificado)
            context_key = str(sorted(obs["context"].items()))
            context_groups[context_key].append(obs)
        
        # Aprender patrones para cada grupo
        for context_key, observations in context_groups.items():
            if len(observations) >= 5:  # Mínimo de observaciones
                pattern = await self._extract_pattern(context_key, observations)
                if pattern:
                    pattern_id = f"pattern_{len(self.patterns)}"
                    self.patterns[pattern_id] = pattern
    
    async def _extract_pattern(
        self,
        context_key: str,
        observations: List[Dict],
    ) -> Optional[LearningPattern]:
        """Extraer patrón de observaciones."""
        # Encontrar acción más común
        actions = [obs["action"] for obs in observations]
        most_common_action = max(set(actions), key=actions.count)
        
        # Calcular tasa de éxito
        successes = sum(
            1 for obs in observations
            if obs["outcome"].get("success", False)
        )
        success_rate = successes / len(observations)
        
        # Calcular confianza basada en número de observaciones
        confidence = min(1.0, len(observations) / 20.0)
        
        # Obtener contexto representativo
        representative_context = observations[0]["context"]
        
        pattern = LearningPattern(
            pattern_id="",
            pattern_type="action_recommendation",
            conditions=representative_context,
            outcome=most_common_action,
            confidence=confidence,
            usage_count=len(observations),
            success_rate=success_rate,
        )
        
        return pattern
    
    async def predict_best_action(
        self,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Predecir mejor acción para contexto dado.
        
        Args:
            context: Contexto actual
        
        Returns:
            Recomendación de acción
        """
        # Buscar patrones que coincidan con el contexto
        matching_patterns = []
        
        for pattern in self.patterns.values():
            if self._context_matches(context, pattern.conditions):
                matching_patterns.append(pattern)
        
        if not matching_patterns:
            return None
        
        # Ordenar por confianza y tasa de éxito
        matching_patterns.sort(
            key=lambda p: (p.confidence * p.success_rate),
            reverse=True,
        )
        
        best_pattern = matching_patterns[0]
        
        return {
            "action": best_pattern.outcome,
            "confidence": best_pattern.confidence,
            "success_rate": best_pattern.success_rate,
            "pattern_id": best_pattern.pattern_id,
        }
    
    def _context_matches(
        self,
        context: Dict[str, Any],
        pattern_conditions: Dict[str, Any],
    ) -> bool:
        """Verificar si el contexto coincide con las condiciones del patrón."""
        # Coincidencia simple (en producción, usar matching más sofisticado)
        for key, value in pattern_conditions.items():
            if key not in context:
                return False
            if isinstance(value, (int, float)):
                # Tolerancia para números
                if abs(context[key] - value) > (value * 0.1):
                    return False
            elif context[key] != value:
                return False
        
        return True
    
    def get_learned_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Obtener patrones aprendidos."""
        patterns = list(self.patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        return [
            {
                "pattern_id": p.pattern_id,
                "pattern_type": p.pattern_type,
                "conditions": p.conditions,
                "outcome": p.outcome,
                "confidence": p.confidence,
                "success_rate": p.success_rate,
                "usage_count": p.usage_count,
            }
            for p in patterns
        ]

