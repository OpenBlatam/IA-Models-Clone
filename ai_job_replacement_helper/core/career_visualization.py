"""
Career Visualization Service - Visualización de trayectoria profesional
========================================================================

Sistema para visualizar y planificar trayectoria profesional.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CareerStage(str, Enum):
    """Etapas de carrera"""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    STAFF = "staff"
    ARCHITECT = "architect"
    MANAGER = "manager"
    DIRECTOR = "director"
    VP = "vp"
    C_LEVEL = "c_level"


@dataclass
class CareerMilestone:
    """Hito de carrera"""
    id: str
    title: str
    company: str
    role: str
    stage: CareerStage
    start_date: datetime
    end_date: Optional[datetime] = None
    achievements: List[str] = field(default_factory=list)
    skills_gained: List[str] = field(default_factory=list)


@dataclass
class CareerPath:
    """Trayectoria profesional"""
    user_id: str
    current_stage: CareerStage
    milestones: List[CareerMilestone]
    target_role: Optional[str] = None
    target_company: Optional[str] = None
    timeline_years: int = 5


@dataclass
class CareerVisualization:
    """Visualización de carrera"""
    current_position: Dict[str, Any]
    path_to_target: List[Dict[str, Any]]
    estimated_timeline: Dict[str, Any]
    required_skills: List[str]
    recommended_steps: List[str]
    growth_rate: float


class CareerVisualizationService:
    """Servicio de visualización de carrera"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.career_paths: Dict[str, CareerPath] = {}
        logger.info("CareerVisualizationService initialized")
    
    def create_career_path(
        self,
        user_id: str,
        current_stage: CareerStage,
        target_role: Optional[str] = None,
        timeline_years: int = 5
    ) -> CareerPath:
        """Crear trayectoria profesional"""
        path = CareerPath(
            user_id=user_id,
            current_stage=current_stage,
            milestones=[],
            target_role=target_role,
            timeline_years=timeline_years,
        )
        
        self.career_paths[user_id] = path
        
        logger.info(f"Career path created for user {user_id}")
        return path
    
    def add_milestone(
        self,
        user_id: str,
        title: str,
        company: str,
        role: str,
        stage: CareerStage,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        achievements: Optional[List[str]] = None
    ) -> CareerMilestone:
        """Agregar hito a la trayectoria"""
        path = self.career_paths.get(user_id)
        if not path:
            raise ValueError(f"Career path not found for user {user_id}")
        
        milestone_id = f"milestone_{user_id}_{int(datetime.now().timestamp())}"
        
        milestone = CareerMilestone(
            id=milestone_id,
            title=title,
            company=company,
            role=role,
            stage=stage,
            start_date=start_date,
            end_date=end_date,
            achievements=achievements or [],
        )
        
        path.milestones.append(milestone)
        
        logger.info(f"Milestone added: {milestone_id}")
        return milestone
    
    def visualize_career_path(
        self,
        user_id: str,
        target_role: Optional[str] = None
    ) -> CareerVisualization:
        """Visualizar trayectoria hacia objetivo"""
        path = self.career_paths.get(user_id)
        if not path:
            raise ValueError(f"Career path not found for user {user_id}")
        
        # Obtener último milestone o posición actual
        current_milestone = path.milestones[-1] if path.milestones else None
        
        current_position = {
            "stage": path.current_stage.value,
            "role": current_milestone.role if current_milestone else "Current Role",
            "company": current_milestone.company if current_milestone else "Current Company",
        }
        
        # Generar path hacia objetivo
        target = target_role or path.target_role
        path_to_target = self._generate_path_to_target(path.current_stage, target)
        
        # Calcular timeline estimado
        estimated_timeline = self._estimate_timeline(path.current_stage, target, path.timeline_years)
        
        # Identificar habilidades requeridas
        required_skills = self._identify_required_skills(path.current_stage, target)
        
        # Generar pasos recomendados
        recommended_steps = self._generate_recommended_steps(path.current_stage, target)
        
        # Calcular tasa de crecimiento
        growth_rate = self._calculate_growth_rate(path.milestones)
        
        return CareerVisualization(
            current_position=current_position,
            path_to_target=path_to_target,
            estimated_timeline=estimated_timeline,
            required_skills=required_skills,
            recommended_steps=recommended_steps,
            growth_rate=growth_rate,
        )
    
    def _generate_path_to_target(
        self,
        current_stage: CareerStage,
        target_role: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generar path hacia objetivo"""
        if not target_role:
            return []
        
        # Mapeo de stages
        stage_order = [
            CareerStage.ENTRY,
            CareerStage.JUNIOR,
            CareerStage.MID,
            CareerStage.SENIOR,
            CareerStage.LEAD,
            CareerStage.PRINCIPAL,
        ]
        
        current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
        
        path = []
        for i in range(current_idx + 1, min(current_idx + 4, len(stage_order))):
            path.append({
                "stage": stage_order[i].value,
                "estimated_years": (i - current_idx) * 1.5,
            })
        
        return path
    
    def _estimate_timeline(
        self,
        current_stage: CareerStage,
        target_role: Optional[str],
        max_years: int
    ) -> Dict[str, Any]:
        """Estimar timeline"""
        if not target_role:
            return {"years": 0, "confidence": 0.0}
        
        # Estimación basada en stage actual
        stage_order = [
            CareerStage.ENTRY,
            CareerStage.JUNIOR,
            CareerStage.MID,
            CareerStage.SENIOR,
        ]
        
        if current_stage in stage_order:
            current_idx = stage_order.index(current_stage)
            years_to_senior = (len(stage_order) - current_idx - 1) * 1.5
        else:
            years_to_senior = 0
        
        return {
            "years": min(years_to_senior, max_years),
            "confidence": 0.7 if years_to_senior <= max_years else 0.4,
            "milestones": [
                {"year": 1, "goal": "Desarrollar habilidades técnicas avanzadas"},
                {"year": 2, "goal": "Liderar proyectos pequeños"},
                {"year": 3, "goal": "Mentorear a otros desarrolladores"},
            ],
        }
    
    def _identify_required_skills(
        self,
        current_stage: CareerStage,
        target_role: Optional[str]
    ) -> List[str]:
        """Identificar habilidades requeridas"""
        if not target_role:
            return []
        
        # Habilidades base según stage
        base_skills = {
            CareerStage.JUNIOR: ["Fundamentos", "Comunicación básica"],
            CareerStage.MID: ["Arquitectura", "Liderazgo técnico"],
            CareerStage.SENIOR: ["Diseño de sistemas", "Mentoring"],
            CareerStage.LEAD: ["Estrategia técnica", "Gestión de equipos"],
        }
        
        return base_skills.get(current_stage, [])
    
    def _generate_recommended_steps(
        self,
        current_stage: CareerStage,
        target_role: Optional[str]
    ) -> List[str]:
        """Generar pasos recomendados"""
        steps = [
            "Identifica las habilidades específicas requeridas para el rol objetivo",
            "Crea un plan de aprendizaje estructurado",
            "Busca proyectos que te permitan desarrollar esas habilidades",
            "Encuentra mentores en el área objetivo",
            "Construye un portafolio que demuestre tus capacidades",
        ]
        
        return steps
    
    def _calculate_growth_rate(self, milestones: List[CareerMilestone]) -> float:
        """Calcular tasa de crecimiento"""
        if len(milestones) < 2:
            return 0.0
        
        # Calcular tiempo entre milestones
        total_days = 0
        for i in range(1, len(milestones)):
            if milestones[i].end_date and milestones[i-1].end_date:
                delta = milestones[i].end_date - milestones[i-1].end_date
                total_days += delta.days
        
        if total_days == 0:
            return 0.0
        
        # Growth rate basado en frecuencia de avances
        avg_days_between = total_days / (len(milestones) - 1)
        growth_rate = 365 / avg_days_between if avg_days_between > 0 else 0.0
        
        return round(growth_rate, 2)




