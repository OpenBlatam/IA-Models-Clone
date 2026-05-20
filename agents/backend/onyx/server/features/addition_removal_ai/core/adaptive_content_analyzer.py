"""
Adaptive Content Analyzer - Sistema de análisis de contenido adaptativo
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AdaptationRule:
    """Regla de adaptación"""
    condition: str
    action: str
    priority: int
    metadata: Dict[str, Any] = None


class AdaptiveContentAnalyzer:
    """Analizador de contenido adaptativo"""

    def __init__(self):
        """Inicializar analizador"""
        self.adaptation_rules: List[AdaptationRule] = []
        self.content_adaptations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_history: Dict[str, List[float]] = defaultdict(list)

    def add_adaptation_rule(
        self,
        condition: str,
        action: str,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Agregar regla de adaptación.

        Args:
            condition: Condición para la adaptación
            action: Acción a realizar
            priority: Prioridad (mayor = más importante)
            metadata: Metadatos adicionales
        """
        rule = AdaptationRule(
            condition=condition,
            action=action,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.adaptation_rules.append(rule)
        # Ordenar por prioridad
        self.adaptation_rules.sort(key=lambda x: x.priority, reverse=True)
        
        logger.debug(f"Regla de adaptación agregada: {condition} -> {action}")

    def analyze_adaptation_needs(
        self,
        content_id: str,
        content_metrics: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analizar necesidades de adaptación.

        Args:
            content_id: ID del contenido
            content_metrics: Métricas del contenido
            user_context: Contexto del usuario (opcional)

        Returns:
            Análisis de necesidades de adaptación
        """
        adaptations = []
        
        # Evaluar reglas de adaptación
        for rule in self.adaptation_rules:
            if self._evaluate_condition(rule.condition, content_metrics, user_context):
                adaptations.append({
                    "rule": rule.condition,
                    "action": rule.action,
                    "priority": rule.priority,
                    "metadata": rule.metadata
                })
        
        # Registrar adaptaciones
        if adaptations:
            self.content_adaptations[content_id].extend(adaptations)
        
        return {
            "content_id": content_id,
            "adaptations_needed": len(adaptations),
            "adaptations": adaptations,
            "timestamp": datetime.utcnow().isoformat()
        }

    def suggest_adaptive_changes(
        self,
        content_id: str,
        current_content: str,
        target_audience: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sugerir cambios adaptativos.

        Args:
            content_id: ID del contenido
            current_content: Contenido actual
            target_audience: Audiencia objetivo (opcional)
            context: Contexto adicional (opcional)

        Returns:
            Sugerencias de cambios adaptativos
        """
        suggestions = []
        
        # Analizar longitud
        word_count = len(current_content.split())
        if word_count > 2000 and target_audience == "mobile":
            suggestions.append({
                "type": "length",
                "priority": "high",
                "issue": "Contenido muy largo para audiencia móvil",
                "suggestion": "Crea una versión resumida o divide en secciones más pequeñas"
            })
        
        # Analizar complejidad
        avg_sentence_length = word_count / max(1, len([s for s in current_content.split('.') if s.strip()]))
        if avg_sentence_length > 25 and target_audience == "beginner":
            suggestions.append({
                "type": "complexity",
                "priority": "high",
                "issue": "Oraciones muy largas para audiencia principiante",
                "suggestion": "Divide las oraciones largas en oraciones más cortas y simples"
            })
        
        # Analizar formato
        if '\n\n' not in current_content and word_count > 500:
            suggestions.append({
                "type": "format",
                "priority": "medium",
                "issue": "Falta de párrafos",
                "suggestion": "Divide el contenido en párrafos para mejorar la legibilidad"
            })
        
        return {
            "content_id": content_id,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "timestamp": datetime.utcnow().isoformat()
        }

    def track_adaptation_performance(
        self,
        content_id: str,
        performance_score: float
    ):
        """
        Rastrear performance de adaptaciones.

        Args:
            content_id: ID del contenido
            performance_score: Score de performance (0-1)
        """
        self.performance_history[content_id].append(performance_score)
        
        # Limitar tamaño
        if len(self.performance_history[content_id]) > 100:
            self.performance_history[content_id] = self.performance_history[content_id][-100:]
        
        logger.debug(f"Performance registrada para {content_id}: {performance_score}")

    def get_adaptation_effectiveness(
        self,
        content_id: str
    ) -> Dict[str, Any]:
        """
        Obtener efectividad de adaptaciones.

        Args:
            content_id: ID del contenido

        Returns:
            Análisis de efectividad
        """
        if content_id not in self.performance_history:
            return {"error": "No hay datos de performance para este contenido"}
        
        performances = self.performance_history[content_id]
        
        if not performances:
            return {"error": "No hay datos de performance"}
        
        avg_performance = sum(performances) / len(performances)
        
        # Comparar performance antes y después de adaptaciones
        adaptations = self.content_adaptations.get(content_id, [])
        if len(performances) >= 2 and adaptations:
            before_adaptations = performances[:len(performances)//2]
            after_adaptations = performances[len(performances)//2:]
            
            before_avg = sum(before_adaptations) / len(before_adaptations)
            after_avg = sum(after_adaptations) / len(after_adaptations)
            
            improvement = after_avg - before_avg
            improvement_percentage = (improvement / before_avg * 100) if before_avg > 0 else 0
        else:
            before_avg = None
            after_avg = None
            improvement = None
            improvement_percentage = None
        
        return {
            "content_id": content_id,
            "average_performance": avg_performance,
            "total_adaptations": len(adaptations),
            "before_adaptations_avg": before_avg,
            "after_adaptations_avg": after_avg,
            "improvement": improvement,
            "improvement_percentage": improvement_percentage,
            "is_effective": improvement > 0.1 if improvement is not None else None
        }

    def _evaluate_condition(
        self,
        condition: str,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluar condición de adaptación"""
        # Implementación simple de evaluación de condiciones
        # En producción, esto podría usar un motor de reglas más sofisticado
        
        try:
            # Reemplazar variables en la condición
            condition_eval = condition
            for key, value in metrics.items():
                condition_eval = condition_eval.replace(f"{{metrics.{key}}}", str(value))
            
            if context:
                for key, value in context.items():
                    condition_eval = condition_eval.replace(f"{{context.{key}}}", str(value))
            
            # Evaluar condición (simple, solo para demostración)
            # En producción, usar un evaluador de expresiones seguro
            return eval(condition_eval, {"__builtins__": {}}, {})
        except Exception as e:
            logger.warning(f"Error evaluando condición {condition}: {e}")
            return False






