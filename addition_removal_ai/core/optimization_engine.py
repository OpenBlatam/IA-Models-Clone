"""
Optimization Engine - Motor de optimización automática
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRule:
    """Regla de optimización"""
    name: str
    condition: callable
    action: callable
    priority: int = 1
    enabled: bool = True


class OptimizationEngine:
    """Motor de optimización automática"""

    def __init__(self):
        """Inicializar motor de optimización"""
        self.rules: List[OptimizationRule] = []
        self.optimizations_applied: List[Dict[str, Any]] = []

    def register_rule(
        self,
        name: str,
        condition: callable,
        action: callable,
        priority: int = 1
    ):
        """
        Registrar regla de optimización.

        Args:
            name: Nombre de la regla
            condition: Función condición
            action: Función acción
            priority: Prioridad
        """
        rule = OptimizationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority
        )
        self.rules.append(rule)
        # Ordenar por prioridad
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Regla de optimización registrada: {name}")

    async def optimize_content(self, content: str) -> Dict[str, Any]:
        """
        Optimizar contenido automáticamente.

        Args:
            content: Contenido a optimizar

        Returns:
            Contenido optimizado y métricas
        """
        optimized_content = content
        applied_optimizations = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                # Verificar condición
                if asyncio.iscoroutinefunction(rule.condition):
                    should_apply = await rule.condition(optimized_content)
                else:
                    should_apply = rule.condition(optimized_content)
                
                if should_apply:
                    # Aplicar acción
                    if asyncio.iscoroutinefunction(rule.action):
                        result = await rule.action(optimized_content)
                    else:
                        result = rule.action(optimized_content)
                    
                    if isinstance(result, str):
                        optimized_content = result
                    elif isinstance(result, dict) and "content" in result:
                        optimized_content = result["content"]
                    
                    applied_optimizations.append({
                        "rule": rule.name,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    logger.info(f"Optimización aplicada: {rule.name}")
            except Exception as e:
                logger.error(f"Error aplicando regla {rule.name}: {e}")
        
        # Registrar optimización
        self.optimizations_applied.append({
            "original_length": len(content),
            "optimized_length": len(optimized_content),
            "optimizations": applied_optimizations,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "content": optimized_content,
            "optimizations_applied": applied_optimizations,
            "improvement": len(content) - len(optimized_content)
        }

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de optimizaciones.

        Returns:
            Estadísticas
        """
        if not self.optimizations_applied:
            return {
                "total_optimizations": 0,
                "avg_improvement": 0.0
            }
        
        total = len(self.optimizations_applied)
        total_improvement = sum(o["improvement"] for o in self.optimizations_applied)
        
        return {
            "total_optimizations": total,
            "avg_improvement": total_improvement / total,
            "total_improvement": total_improvement
        }

