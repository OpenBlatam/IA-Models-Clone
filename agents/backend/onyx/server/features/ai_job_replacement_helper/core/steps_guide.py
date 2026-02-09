"""
Steps Guide Service - Sistema de pasos guiados
==============================================

Sistema de pasos personalizados y roadmap para ayudar a los usuarios
cuando una IA les quita su trabajo.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Estado de un paso"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class StepCategory(str, Enum):
    """Categorías de pasos"""
    ASSESSMENT = "assessment"
    SKILL_DEVELOPMENT = "skill_development"
    NETWORKING = "networking"
    JOB_SEARCH = "job_search"
    APPLICATION = "application"
    INTERVIEW = "interview"
    NEGOTIATION = "negotiation"
    MINDSET = "mindset"


@dataclass
class Step:
    """Representa un paso en el roadmap"""
    id: str
    title: str
    description: str
    category: StepCategory
    order: int
    estimated_time: str  # "30 min", "2 hours", "1 day"
    resources: List[Dict[str, str]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None


class StepsGuideService:
    """Servicio de pasos guiados"""
    
    def __init__(self):
        """Inicializar servicio de pasos"""
        self.user_steps: Dict[str, List[Step]] = {}
        logger.info("StepsGuideService initialized")
    
    def get_default_roadmap(self) -> List[Step]:
        """Obtener roadmap por defecto"""
        return [
            Step(
                id="step_1",
                title="Evalúa tu situación actual",
                description="Analiza qué habilidades tienes, qué te gusta hacer y qué oportunidades hay en el mercado",
                category=StepCategory.ASSESSMENT,
                order=1,
                estimated_time="1 hour",
                resources=[
                    {"type": "article", "title": "Cómo hacer un auto-análisis profesional", "url": "#"},
                    {"type": "tool", "title": "Test de habilidades", "url": "#"},
                ]
            ),
            Step(
                id="step_2",
                title="Identifica nuevas habilidades demandadas",
                description="Investiga qué habilidades están en demanda y cuáles se alinean con tus intereses",
                category=StepCategory.ASSESSMENT,
                order=2,
                estimated_time="2 hours",
                resources=[
                    {"type": "article", "title": "Top 10 habilidades más demandadas 2024", "url": "#"},
                    {"type": "video", "title": "Cómo identificar oportunidades", "url": "#"},
                ]
            ),
            Step(
                id="step_3",
                title="Crea un plan de aprendizaje",
                description="Diseña un plan estructurado para aprender las nuevas habilidades",
                category=StepCategory.SKILL_DEVELOPMENT,
                order=3,
                estimated_time="1 hour",
                prerequisites=["step_1", "step_2"],
                resources=[
                    {"type": "template", "title": "Plantilla de plan de aprendizaje", "url": "#"},
                ]
            ),
            Step(
                id="step_4",
                title="Actualiza tu perfil de LinkedIn",
                description="Optimiza tu perfil para que sea atractivo para reclutadores y algoritmos",
                category=StepCategory.NETWORKING,
                order=4,
                estimated_time="2 hours",
                prerequisites=["step_1"],
                resources=[
                    {"type": "guide", "title": "Guía completa de LinkedIn", "url": "#"},
                    {"type": "checklist", "title": "Checklist de perfil optimizado", "url": "#"},
                ]
            ),
            Step(
                id="step_5",
                title="Construye tu red profesional",
                description="Conecta con profesionales en tu nueva área de interés",
                category=StepCategory.NETWORKING,
                order=5,
                estimated_time="Ongoing",
                prerequisites=["step_4"],
                resources=[
                    {"type": "article", "title": "Cómo hacer networking efectivo", "url": "#"},
                ]
            ),
            Step(
                id="step_6",
                title="Busca oportunidades de trabajo",
                description="Usa nuestra herramienta estilo Tinder para encontrar trabajos que te interesen",
                category=StepCategory.JOB_SEARCH,
                order=6,
                estimated_time="Ongoing",
                prerequisites=["step_4"],
            ),
            Step(
                id="step_7",
                title="Prepara tu CV y carta de presentación",
                description="Crea documentos que destaquen tus nuevas habilidades y experiencia",
                category=StepCategory.APPLICATION,
                order=7,
                estimated_time="4 hours",
                prerequisites=["step_3"],
                resources=[
                    {"type": "template", "title": "Plantillas de CV", "url": "#"},
                    {"type": "tool", "title": "Generador de cartas de presentación", "url": "#"},
                ]
            ),
            Step(
                id="step_8",
                title="Practica para entrevistas",
                description="Prepárate para entrevistas con práctica y simulaciones",
                category=StepCategory.INTERVIEW,
                order=8,
                estimated_time="Ongoing",
                prerequisites=["step_7"],
                resources=[
                    {"type": "video", "title": "Preguntas comunes de entrevista", "url": "#"},
                    {"type": "tool", "title": "Simulador de entrevistas con IA", "url": "#"},
                ]
            ),
            Step(
                id="step_9",
                title="Aplica a trabajos",
                description="Envía aplicaciones a las oportunidades que más te interesen",
                category=StepCategory.APPLICATION,
                order=9,
                estimated_time="Ongoing",
                prerequisites=["step_6", "step_7"],
            ),
            Step(
                id="step_10",
                title="Mantén una mentalidad positiva",
                description="El proceso puede ser largo, mantén la motivación y aprende de cada experiencia",
                category=StepCategory.MINDSET,
                order=10,
                estimated_time="Ongoing",
                resources=[
                    {"type": "article", "title": "Cómo mantener la motivación", "url": "#"},
                    {"type": "community", "title": "Grupo de apoyo", "url": "#"},
                ]
            ),
        ]
    
    def get_user_roadmap(self, user_id: str) -> List[Step]:
        """Obtener roadmap del usuario"""
        if user_id not in self.user_steps:
            self.user_steps[user_id] = self.get_default_roadmap()
        return self.user_steps[user_id]
    
    def start_step(self, user_id: str, step_id: str) -> Dict[str, Any]:
        """Iniciar un paso"""
        roadmap = self.get_user_roadmap(user_id)
        step = next((s for s in roadmap if s.id == step_id), None)
        
        if not step:
            raise ValueError(f"Step {step_id} not found")
        
        # Verificar prerrequisitos
        if step.prerequisites:
            for prereq_id in step.prerequisites:
                prereq = next((s for s in roadmap if s.id == prereq_id), None)
                if prereq and prereq.status != StepStatus.COMPLETED:
                    return {
                        "success": False,
                        "message": f"Debes completar primero: {prereq.title}",
                        "required_step": prereq_id,
                    }
        
        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.now()
        
        return {
            "success": True,
            "step": self._step_to_dict(step),
        }
    
    def complete_step(self, user_id: str, step_id: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """Completar un paso"""
        roadmap = self.get_user_roadmap(user_id)
        step = next((s for s in roadmap if s.id == step_id), None)
        
        if not step:
            raise ValueError(f"Step {step_id} not found")
        
        step.status = StepStatus.COMPLETED
        step.completed_at = datetime.now()
        if notes:
            step.notes = notes
        
        # Desbloquear siguientes pasos
        next_steps = [
            s for s in roadmap
            if step_id in s.prerequisites and s.status == StepStatus.NOT_STARTED
        ]
        
        return {
            "success": True,
            "step": self._step_to_dict(step),
            "unlocked_steps": [self._step_to_dict(s) for s in next_steps],
        }
    
    def get_step_progress(self, user_id: str) -> Dict[str, Any]:
        """Obtener progreso de pasos del usuario"""
        roadmap = self.get_user_roadmap(user_id)
        
        total_steps = len(roadmap)
        completed_steps = sum(1 for s in roadmap if s.status == StepStatus.COMPLETED)
        in_progress_steps = sum(1 for s in roadmap if s.status == StepStatus.IN_PROGRESS)
        not_started_steps = sum(1 for s in roadmap if s.status == StepStatus.NOT_STARTED)
        
        progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Agrupar por categoría
        by_category = {}
        for category in StepCategory:
            category_steps = [s for s in roadmap if s.category == category]
            by_category[category.value] = {
                "total": len(category_steps),
                "completed": sum(1 for s in category_steps if s.status == StepStatus.COMPLETED),
                "in_progress": sum(1 for s in category_steps if s.status == StepStatus.IN_PROGRESS),
            }
        
        return {
            "total_steps": total_steps,
            "completed": completed_steps,
            "in_progress": in_progress_steps,
            "not_started": not_started_steps,
            "progress_percentage": round(progress_percentage, 2),
            "by_category": by_category,
            "steps": [self._step_to_dict(s) for s in roadmap],
        }
    
    def _step_to_dict(self, step: Step) -> Dict[str, Any]:
        """Convertir Step a diccionario"""
        return {
            "id": step.id,
            "title": step.title,
            "description": step.description,
            "category": step.category.value,
            "order": step.order,
            "estimated_time": step.estimated_time,
            "status": step.status.value,
            "resources": step.resources,
            "prerequisites": step.prerequisites,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "completed_at": step.completed_at.isoformat() if step.completed_at else None,
            "notes": step.notes,
        }




