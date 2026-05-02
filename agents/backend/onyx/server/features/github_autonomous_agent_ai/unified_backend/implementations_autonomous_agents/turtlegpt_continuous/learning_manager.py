"""
Learning Manager Module
=======================

Gestiona el aprendizaje autónomo basado en Self-Initiated Learning paper.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LearningManager:
    """
    Gestiona oportunidades de aprendizaje autónomo.
    
    Basado en el paper "Self-Initiated Learning":
    - Detecta tareas novedosas
    - Identifica caídas de rendimiento
    - Reconoce errores frecuentes
    - Activa curiosidad sobre conceptos desconocidos
    """
    
    def __init__(
        self,
        learning_enabled: bool = True,
        novelty_threshold: float = 0.0,
        performance_threshold: float = 0.6,
        error_threshold: int = 3
    ):
        """
        Inicializar gestor de aprendizaje.
        
        Args:
            learning_enabled: Si el aprendizaje está habilitado
            novelty_threshold: Umbral para detectar novedad
            performance_threshold: Umbral de rendimiento mínimo
            error_threshold: Número de errores para activar aprendizaje
        """
        self.learning_enabled = learning_enabled
        self.novelty_threshold = novelty_threshold
        self.performance_threshold = performance_threshold
        self.error_threshold = error_threshold
        self.learning_opportunities: List[Dict[str, Any]] = []
        self.learned_concepts: List[str] = []
    
    def identify_opportunities(
        self,
        recent_tasks: List[Any],
        semantic_memory: Any,
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identificar oportunidades de aprendizaje.
        
        Args:
            recent_tasks: Lista de tareas recientes
            semantic_memory: Memoria semántica para verificar novedad
            metrics: Métricas del agente
            
        Returns:
            Lista de oportunidades de aprendizaje identificadas
        """
        if not self.learning_enabled:
            return []
        
        opportunities = []
        
        # 1. Detectar tareas novedosas
        novel_opportunities = self._detect_novel_tasks(recent_tasks, semantic_memory)
        opportunities.extend(novel_opportunities)
        
        # 2. Detectar caídas de rendimiento
        performance_opportunities = self._detect_performance_drops(metrics)
        opportunities.extend(performance_opportunities)
        
        # 3. Detectar errores frecuentes
        error_opportunities = self._detect_frequent_errors(metrics)
        opportunities.extend(error_opportunities)
        
        # Guardar oportunidades
        self.learning_opportunities.extend(opportunities)
        
        return opportunities
    
    def _detect_novel_tasks(
        self,
        recent_tasks: List[Any],
        semantic_memory: Any
    ) -> List[Dict[str, Any]]:
        """Detectar tareas novedosas."""
        opportunities = []
        
        for task in recent_tasks[:5]:  # Revisar últimas 5 tareas
            # Verificar si es novedosa (no está en memoria semántica)
            try:
                similar = semantic_memory.query(pattern=task.description)
                if len(similar) == 0:
                    opportunities.append({
                        "type": "novel_task",
                        "description": f"Novel task encountered: {task.description}",
                        "priority": 0.8,
                        "trigger": "curiosity",
                        "timestamp": datetime.now().isoformat(),
                        "task_id": getattr(task, "task_id", None)
                    })
            except Exception as e:
                logger.warning(f"Error checking novelty: {e}")
        
        return opportunities
    
    def _detect_performance_drops(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detectar caídas de rendimiento."""
        opportunities = []
        
        total_processed = metrics.get("total_tasks_processed", 0)
        if total_processed > 5:
            success_rate = metrics.get("success_rate", 0.0)
            if success_rate < self.performance_threshold:
                opportunities.append({
                    "type": "performance_drop",
                    "description": f"Performance dropped to {success_rate:.2%}",
                    "priority": 0.9,
                    "trigger": "performance_drop",
                    "timestamp": datetime.now().isoformat(),
                    "current_rate": success_rate,
                    "threshold": self.performance_threshold
                })
        
        return opportunities
    
    def _detect_frequent_errors(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detectar errores frecuentes."""
        opportunities = []
        
        error_count = metrics.get("errors_count", 0)
        if error_count >= self.error_threshold:
            opportunities.append({
                "type": "frequent_errors",
                "description": f"Multiple errors detected: {error_count}",
                "priority": 0.85,
                "trigger": "error_detection",
                "timestamp": datetime.now().isoformat(),
                "error_count": error_count,
                "threshold": self.error_threshold
            })
        
        return opportunities
    
    def record_learning(self, concept: str, success: bool = True):
        """
        Registrar un concepto aprendido.
        
        Args:
            concept: Concepto aprendido
            success: Si el aprendizaje fue exitoso
        """
        if success and concept not in self.learned_concepts:
            self.learned_concepts.append(concept)
            logger.info(f"Learned new concept: {concept}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de aprendizaje."""
        return {
            "learning_enabled": self.learning_enabled,
            "opportunities_count": len(self.learning_opportunities),
            "learned_concepts_count": len(self.learned_concepts),
            "recent_opportunities": self.learning_opportunities[-10:],  # Últimas 10
            "learned_concepts": self.learned_concepts[-20:]  # Últimos 20 conceptos
        }
