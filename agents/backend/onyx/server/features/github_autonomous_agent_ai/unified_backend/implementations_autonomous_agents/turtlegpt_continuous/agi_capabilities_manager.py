"""
AGI Capabilities Manager
========================

Gestiona y evalúa las capacidades tipo AGI del agente.
Basado en "Sparks of Artificial General Intelligence" paper.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AGICapabilitiesManager:
    """
    Gestiona capacidades tipo AGI del agente.
    
    Basado en el paper "Sparks of AGI":
    - Razonamiento avanzado
    - Planificación compleja
    - Uso de herramientas
    - Resolución de problemas
    - Adaptación y aprendizaje
    """
    
    def __init__(self):
        """Inicializar gestor de capacidades AGI."""
        self.capabilities: Dict[str, float] = {
            "reasoning": 0.7,
            "planning": 0.7,
            "tool_use": 0.8,
            "problem_solving": 0.7,
            "adaptation": 0.6,
            "learning": 0.6,
            "creativity": 0.5,
            "multi_modal": 0.4
        }
        self.capability_history: Dict[str, list] = {
            capability: [] for capability in self.capabilities.keys()
        }
        self.evaluation_events: List[Dict[str, Any]] = []
    
    def update_capability(
        self,
        capability: str,
        score: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Actualizar score de una capacidad.
        
        Args:
            capability: Nombre de la capacidad
            score: Nuevo score (0.0 a 1.0)
            context: Contexto adicional
        """
        if capability not in self.capabilities:
            logger.warning(f"Unknown capability: {capability}")
            return
        
        # Actualizar score (promedio móvil)
        old_score = self.capabilities[capability]
        self.capabilities[capability] = (old_score * 0.7 + score * 0.3)
        
        # Guardar en historial
        self.capability_history[capability].append({
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        })
        
        # Mantener solo últimos 100 eventos
        if len(self.capability_history[capability]) > 100:
            self.capability_history[capability] = self.capability_history[capability][-100:]
        
        logger.debug(f"Updated {capability} capability: {old_score:.2f} -> {self.capabilities[capability]:.2f}")
    
    def evaluate_reasoning(self, task_complexity: float, success: bool) -> float:
        """
        Evaluar capacidad de razonamiento.
        
        Args:
            task_complexity: Complejidad de la tarea (0.0 a 1.0)
            success: Si la tarea fue exitosa
            
        Returns:
            Score de razonamiento
        """
        score = 0.5
        if success:
            score = 0.5 + (task_complexity * 0.5)
        else:
            score = 0.3 - (task_complexity * 0.2)
        
        self.update_capability("reasoning", max(0.0, min(1.0, score)))
        return score
    
    def evaluate_planning(self, plan_quality: float, execution_success: bool) -> float:
        """
        Evaluar capacidad de planificación.
        
        Args:
            plan_quality: Calidad del plan (0.0 a 1.0)
            execution_success: Si la ejecución fue exitosa
            
        Returns:
            Score de planificación
        """
        score = plan_quality * 0.7 + (0.3 if execution_success else 0.0)
        self.update_capability("planning", max(0.0, min(1.0, score)))
        return score
    
    def evaluate_tool_use(
        self,
        tool_used: bool,
        tool_effectiveness: float
    ) -> float:
        """
        Evaluar capacidad de uso de herramientas.
        
        Args:
            tool_used: Si se usó una herramienta
            tool_effectiveness: Efectividad del uso (0.0 a 1.0)
            
        Returns:
            Score de uso de herramientas
        """
        if not tool_used:
            score = 0.3
        else:
            score = 0.5 + (tool_effectiveness * 0.5)
        
        self.update_capability("tool_use", max(0.0, min(1.0, score)))
        return score
    
    def evaluate_problem_solving(
        self,
        problem_solved: bool,
        solution_quality: float
    ) -> float:
        """
        Evaluar capacidad de resolución de problemas.
        
        Args:
            problem_solved: Si el problema fue resuelto
            solution_quality: Calidad de la solución (0.0 a 1.0)
            
        Returns:
            Score de resolución de problemas
        """
        if problem_solved:
            score = 0.6 + (solution_quality * 0.4)
        else:
            score = 0.2
        
        self.update_capability("problem_solving", max(0.0, min(1.0, score)))
        return score
    
    def get_overall_agi_score(self) -> float:
        """
        Calcular score general de AGI.
        
        Returns:
            Score promedio de todas las capacidades
        """
        scores = list(self.capabilities.values())
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_capabilities_report(self) -> Dict[str, Any]:
        """
        Obtener reporte completo de capacidades.
        
        Returns:
            Dict con información detallada de capacidades
        """
        return {
            "capabilities": self.capabilities.copy(),
            "overall_score": self.get_overall_agi_score(),
            "capability_trends": {
                cap: {
                    "current": self.capabilities[cap],
                    "recent_avg": (
                        sum([e["score"] for e in self.capability_history[cap][-10:]]) / 
                        len(self.capability_history[cap][-10:])
                        if len(self.capability_history[cap]) > 0 else 0.0
                    ),
                    "history_size": len(self.capability_history[cap])
                }
                for cap in self.capabilities.keys()
            },
            "top_capabilities": sorted(
                self.capabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            "improving_capabilities": self._identify_improving_capabilities()
        }
    
    def _identify_improving_capabilities(self) -> List[str]:
        """Identificar capacidades que están mejorando."""
        improving = []
        
        for capability in self.capabilities.keys():
            history = self.capability_history[capability]
            if len(history) >= 5:
                recent_avg = sum([e["score"] for e in history[-5:]]) / 5
                older_avg = sum([e["score"] for e in history[-10:-5]]) / 5 if len(history) >= 10 else recent_avg
                
                if recent_avg > older_avg + 0.05:  # Mejora significativa
                    improving.append(capability)
        
        return improving
