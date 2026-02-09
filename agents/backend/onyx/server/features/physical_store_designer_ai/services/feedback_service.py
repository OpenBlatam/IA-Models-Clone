"""
Feedback Service - Sistema de feedback y iteración
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..core.models import StoreDesign

logger = logging.getLogger(__name__)


class FeedbackService:
    """Servicio para manejar feedback y iteraciones"""
    
    def __init__(self):
        self.feedback_store: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_feedback(
        self,
        store_id: str,
        feedback_type: str,
        content: str,
        rating: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Agregar feedback a un diseño"""
        
        feedback = {
            "id": f"{store_id}_{len(self.feedback_store.get(store_id, [])) + 1}",
            "store_id": store_id,
            "type": feedback_type,  # "general", "layout", "decoration", "marketing", "financial"
            "content": content,
            "rating": rating,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"  # "pending", "addressed", "rejected"
        }
        
        if store_id not in self.feedback_store:
            self.feedback_store[store_id] = []
        
        self.feedback_store[store_id].append(feedback)
        
        logger.info(f"Feedback agregado para diseño {store_id}")
        return feedback
    
    def get_feedback(self, store_id: str) -> List[Dict[str, Any]]:
        """Obtener todo el feedback de un diseño"""
        return self.feedback_store.get(store_id, [])
    
    def update_feedback_status(
        self,
        store_id: str,
        feedback_id: str,
        status: str
    ) -> bool:
        """Actualizar estado de feedback"""
        if store_id not in self.feedback_store:
            return False
        
        for feedback in self.feedback_store[store_id]:
            if feedback["id"] == feedback_id:
                feedback["status"] = status
                feedback["updated_at"] = datetime.now().isoformat()
                return True
        
        return False
    
    def generate_improvement_suggestions(
        self,
        design: StoreDesign,
        feedback_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generar sugerencias de mejora basadas en feedback"""
        
        if not feedback_list:
            return {
                "suggestions": [],
                "summary": "No hay feedback disponible"
            }
        
        suggestions = []
        
        # Analizar feedback por tipo
        feedback_by_type = {}
        for feedback in feedback_list:
            ftype = feedback.get("type", "general")
            if ftype not in feedback_by_type:
                feedback_by_type[ftype] = []
            feedback_by_type[ftype].append(feedback)
        
        # Sugerencias para layout
        if "layout" in feedback_by_type:
            layout_feedback = feedback_by_type["layout"]
            suggestions.append({
                "area": "Layout",
                "suggestions": [
                    "Revisar distribución de zonas según feedback",
                    "Considerar flujo de tráfico mejorado",
                    "Optimizar uso del espacio"
                ],
                "priority": "high" if len(layout_feedback) > 2 else "medium"
            })
        
        # Sugerencias para decoración
        if "decoration" in feedback_by_type:
            deco_feedback = feedback_by_type["decoration"]
            suggestions.append({
                "area": "Decoración",
                "suggestions": [
                    "Ajustar esquema de colores si es necesario",
                    "Revisar selección de muebles",
                    "Considerar elementos decorativos adicionales"
                ],
                "priority": "medium"
            })
        
        # Sugerencias para marketing
        if "marketing" in feedback_by_type:
            marketing_feedback = feedback_by_type["marketing"]
            suggestions.append({
                "area": "Marketing",
                "suggestions": [
                    "Refinar estrategias de marketing",
                    "Agregar más tácticas de ventas",
                    "Mejorar plan de redes sociales"
                ],
                "priority": "medium"
            })
        
        # Sugerencias financieras
        if "financial" in feedback_by_type:
            financial_feedback = feedback_by_type["financial"]
            suggestions.append({
                "area": "Finanzas",
                "suggestions": [
                    "Revisar proyecciones financieras",
                    "Optimizar costos iniciales",
                    "Mejorar estrategia de precios"
                ],
                "priority": "high"
            })
        
        # Calcular rating promedio
        ratings = [f.get("rating") for f in feedback_list if f.get("rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        return {
            "suggestions": suggestions,
            "summary": {
                "total_feedback": len(feedback_list),
                "average_rating": round(avg_rating, 1) if avg_rating else None,
                "feedback_by_type": {k: len(v) for k, v in feedback_by_type.items()}
            },
            "next_steps": self._generate_next_steps(suggestions)
        }
    
    def _generate_next_steps(self, suggestions: List[Dict[str, Any]]) -> List[str]:
        """Generar próximos pasos basados en sugerencias"""
        next_steps = []
        
        high_priority = [s for s in suggestions if s.get("priority") == "high"]
        if high_priority:
            next_steps.append("Priorizar mejoras de alta prioridad")
            next_steps.append("Revisar diseño completo con feedback")
        
        if len(suggestions) > 3:
            next_steps.append("Considerar crear versión mejorada del diseño")
        
        next_steps.append("Implementar mejoras sugeridas")
        next_steps.append("Solicitar feedback adicional después de mejoras")
        
        return next_steps
    
    def create_design_version(
        self,
        original_design: StoreDesign,
        improvements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crear versión mejorada del diseño"""
        return {
            "original_store_id": original_design.store_id,
            "version": "2.0",
            "improvements": improvements,
            "created_at": datetime.now().isoformat(),
            "notes": "Versión mejorada basada en feedback"
        }




