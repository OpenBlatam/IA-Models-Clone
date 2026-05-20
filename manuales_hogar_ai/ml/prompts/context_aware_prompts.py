"""
Context-Aware Prompts
=====================

Prompts que se adaptan al contexto y mejoran con el tiempo.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ContextAwarePrompts:
    """Prompts conscientes del contexto."""
    
    def __init__(self):
        """Inicializar sistema de prompts."""
        self.prompt_history: List[Dict[str, Any]] = []
        self.prompt_metrics: Dict[str, Dict[str, float]] = {}
        self._logger = logger
    
    def build_adaptive_prompt(
        self,
        problem_description: str,
        category: str,
        user_feedback: Optional[Dict[str, Any]] = None,
        similar_manuals: Optional[List[Dict[str, Any]]] = None,
        difficulty: Optional[str] = None
    ) -> str:
        """
        Construir prompt adaptativo.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría
            user_feedback: Feedback de usuario
            similar_manuals: Manuales similares
            difficulty: Dificultad estimada
        
        Returns:
            Prompt adaptativo
        """
        # Base prompt
        prompt_parts = [
            f"Genera un manual paso a paso tipo LEGO para {category}.",
            "",
            f"PROBLEMA: {problem_description}",
            ""
        ]
        
        # Agregar contexto de dificultad
        if difficulty:
            prompt_parts.append(f"DIFICULTAD ESTIMADA: {difficulty}")
            prompt_parts.append("")
        
        # Agregar manuales similares
        if similar_manuals:
            prompt_parts.append("MANUALES SIMILARES DE REFERENCIA:")
            for i, manual in enumerate(similar_manuals[:3], 1):
                prompt_parts.append(f"\n--- Referencia {i} ---")
                prompt_parts.append(f"Problema: {manual.get('problem', '')}")
                prompt_parts.append(f"Solución: {manual.get('content', '')[:300]}")
            prompt_parts.append("")
        
        # Agregar feedback si existe
        if user_feedback:
            if user_feedback.get('previous_issues'):
                prompt_parts.append("PROBLEMAS ANTERIORES A EVITAR:")
                for issue in user_feedback['previous_issues']:
                    prompt_parts.append(f"- {issue}")
                prompt_parts.append("")
        
        # Instrucciones adaptativas
        prompt_parts.extend([
            "INSTRUCCIONES ESPECÍFICAS:",
            "- Formato paso a paso tipo LEGO, muy visual",
            "- Cada paso debe ser claro y accionable",
            "- Incluir advertencias de seguridad relevantes",
            "- Listar herramientas específicas necesarias",
            "- Listar materiales con cantidades aproximadas",
            "- Proporcionar estimación de tiempo realista",
            "- Incluir tips y trucos útiles",
            "- Mencionar problemas comunes y cómo evitarlos",
            "",
            "MANUAL:"
        ])
        
        return "\n".join(prompt_parts)
    
    def learn_from_feedback(
        self,
        prompt: str,
        feedback: Dict[str, Any]
    ):
        """
        Aprender de feedback para mejorar prompts.
        
        Args:
            prompt: Prompt usado
            feedback: Feedback del usuario
        """
        try:
            # Guardar en historial
            self.prompt_history.append({
                "prompt": prompt,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            
            # Actualizar métricas
            prompt_hash = hash(prompt)
            if prompt_hash not in self.prompt_metrics:
                self.prompt_metrics[prompt_hash] = {
                    "total": 0,
                    "positive": 0,
                    "negative": 0,
                    "avg_rating": 0.0
                }
            
            metrics = self.prompt_metrics[prompt_hash]
            metrics["total"] += 1
            
            if feedback.get("rating", 0) >= 4:
                metrics["positive"] += 1
            elif feedback.get("rating", 0) <= 2:
                metrics["negative"] += 1
            
            if "rating" in feedback:
                current_avg = metrics["avg_rating"]
                new_rating = feedback["rating"]
                metrics["avg_rating"] = (
                    (current_avg * (metrics["total"] - 1) + new_rating) / metrics["total"]
                )
            
            self._logger.info(f"Métricas actualizadas para prompt: {metrics}")
        
        except Exception as e:
            self._logger.error(f"Error aprendiendo de feedback: {str(e)}")
    
    def get_best_prompt_pattern(self) -> Optional[str]:
        """Obtener patrón de prompt con mejor rendimiento."""
        if not self.prompt_metrics:
            return None
        
        best_hash = max(
            self.prompt_metrics.items(),
            key=lambda x: x[1]["avg_rating"]
        )[0]
        
        # Buscar en historial
        for entry in self.prompt_history:
            if hash(entry["prompt"]) == best_hash:
                return entry["prompt"]
        
        return None




