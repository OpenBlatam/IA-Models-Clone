"""
Continuous Learning System
===========================

Sistema de aprendizaje continuo.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..analytics.analytics import get_analytics_engine
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """Patrón de aprendizaje."""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[str] = None


class ContinuousLearningSystem:
    """
    Sistema de aprendizaje continuo.
    
    Aprende de las operaciones y mejora automáticamente.
    """
    
    def __init__(self):
        """Inicializar sistema de aprendizaje continuo."""
        self.patterns: Dict[str, LearningPattern] = {}
        self.learning_history: List[Dict[str, Any]] = []
        self.enabled = True
        self.min_success_rate = 0.7
    
    def learn_from_operation(
        self,
        operation_type: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
        success: bool
    ) -> None:
        """
        Aprender de una operación.
        
        Args:
            operation_type: Tipo de operación
            input_data: Datos de entrada
            result: Resultado
            success: Si fue exitosa
        """
        if not self.enabled:
            return
        
        # Identificar patrón
        pattern_id = self._identify_pattern(operation_type, input_data)
        
        if pattern_id not in self.patterns:
            # Crear nuevo patrón
            pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type=operation_type,
                conditions=input_data,
                actions=result,
                success_rate=1.0 if success else 0.0,
                usage_count=1,
                last_used=datetime.now().isoformat()
            )
            self.patterns[pattern_id] = pattern
        else:
            # Actualizar patrón existente
            pattern = self.patterns[pattern_id]
            pattern.usage_count += 1
            pattern.last_used = datetime.now().isoformat()
            
            # Actualizar success rate
            total_success = pattern.success_rate * (pattern.usage_count - 1)
            if success:
                total_success += 1
            pattern.success_rate = total_success / pattern.usage_count
        
        # Registrar en historial
        self.learning_history.append({
            "pattern_id": pattern_id,
            "operation_type": operation_type,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    def _identify_pattern(
        self,
        operation_type: str,
        input_data: Dict[str, Any]
    ) -> str:
        """
        Identificar patrón de operación.
        
        Args:
            operation_type: Tipo de operación
            input_data: Datos de entrada
            
        Returns:
            ID del patrón
        """
        # Crear ID basado en tipo y características clave
        key_features = []
        
        if "algorithm" in input_data:
            key_features.append(f"alg_{input_data['algorithm']}")
        
        if "obstacles" in input_data:
            obstacle_count = len(input_data["obstacles"]) if input_data["obstacles"] else 0
            key_features.append(f"obs_{obstacle_count}")
        
        if "distance" in input_data:
            distance = input_data["distance"]
            if distance < 0.5:
                key_features.append("dist_short")
            elif distance < 2.0:
                key_features.append("dist_medium")
            else:
                key_features.append("dist_long")
        
        pattern_id = f"{operation_type}_{'_'.join(key_features)}"
        return pattern_id
    
    def get_recommendation(
        self,
        operation_type: str,
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Obtener recomendación basada en aprendizaje.
        
        Args:
            operation_type: Tipo de operación
            input_data: Datos de entrada
            
        Returns:
            Recomendación o None
        """
        if not self.enabled:
            return None
        
        # Identificar patrón
        pattern_id = self._identify_pattern(operation_type, input_data)
        
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            # Solo recomendar si tiene buena tasa de éxito
            if pattern.success_rate >= self.min_success_rate:
                return {
                    "pattern_id": pattern_id,
                    "recommended_actions": pattern.actions,
                    "confidence": pattern.success_rate,
                    "usage_count": pattern.usage_count
                }
        
        return None
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de aprendizaje."""
        total_patterns = len(self.patterns)
        high_success_patterns = sum(
            1 for p in self.patterns.values()
            if p.success_rate >= self.min_success_rate
        )
        
        total_operations = sum(p.usage_count for p in self.patterns.values())
        
        return {
            "total_patterns": total_patterns,
            "high_success_patterns": high_success_patterns,
            "total_operations_learned": total_operations,
            "average_success_rate": (
                sum(p.success_rate for p in self.patterns.values()) / total_patterns
                if total_patterns > 0 else 0.0
            ),
            "enabled": self.enabled
        }
    
    def export_patterns(self, filepath: str) -> None:
        """
        Exportar patrones aprendidos.
        
        Args:
            filepath: Ruta del archivo
        """
        try:
            patterns_data = {
                pattern_id: {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "conditions": pattern.conditions,
                    "actions": pattern.actions,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "last_used": pattern.last_used
                }
                for pattern_id, pattern in self.patterns.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            logger.info(f"Patterns exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
    
    def import_patterns(self, filepath: str) -> None:
        """
        Importar patrones aprendidos.
        
        Args:
            filepath: Ruta del archivo
        """
        try:
            with open(filepath, 'r') as f:
                patterns_data = json.load(f)
            
            for pattern_id, data in patterns_data.items():
                pattern = LearningPattern(
                    pattern_id=data["pattern_id"],
                    pattern_type=data["pattern_type"],
                    conditions=data["conditions"],
                    actions=data["actions"],
                    success_rate=data["success_rate"],
                    usage_count=data["usage_count"],
                    last_used=data.get("last_used")
                )
                self.patterns[pattern_id] = pattern
            
            logger.info(f"Patterns imported from {filepath}")
        except Exception as e:
            logger.error(f"Error importing patterns: {e}")


# Instancia global
_continuous_learning: Optional[ContinuousLearningSystem] = None


def get_continuous_learning() -> ContinuousLearningSystem:
    """Obtener instancia global del sistema de aprendizaje continuo."""
    global _continuous_learning
    if _continuous_learning is None:
        _continuous_learning = ContinuousLearningSystem()
    return _continuous_learning






