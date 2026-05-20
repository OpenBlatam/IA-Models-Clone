"""
Advanced Skill Gap Analysis Service - Análisis avanzado de brechas de habilidades
==================================================================================

Sistema de análisis profundo de brechas de habilidades con recomendaciones personalizadas.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SkillGap:
    """Brecha de habilidad"""
    skill: str
    current_level: float  # 0.0 - 1.0
    required_level: float  # 0.0 - 1.0
    gap_size: float
    priority: str  # "critical", "high", "medium", "low"
    learning_path: List[str]
    estimated_time: int  # días
    resources: List[Dict[str, str]]


@dataclass
class SkillGapAnalysis:
    """Análisis completo de brechas"""
    user_id: str
    target_role: str
    current_skills: Dict[str, float]
    required_skills: Dict[str, float]
    gaps: List[SkillGap]
    overall_gap_score: float
    readiness_score: float
    recommendations: List[str]
    learning_plan: Dict[str, Any]


class AdvancedSkillGapService:
    """Servicio de análisis avanzado de brechas"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.analyses: Dict[str, SkillGapAnalysis] = {}
        logger.info("AdvancedSkillGapService initialized")
    
    def analyze_skill_gaps(
        self,
        user_id: str,
        target_role: str,
        current_skills: Dict[str, float],
        required_skills: Optional[Dict[str, float]] = None
    ) -> SkillGapAnalysis:
        """Analizar brechas de habilidades"""
        # Si no se proporcionan habilidades requeridas, obtenerlas del rol objetivo
        if not required_skills:
            required_skills = self._get_required_skills_for_role(target_role)
        
        # Identificar brechas
        gaps = []
        for skill, required_level in required_skills.items():
            current_level = current_skills.get(skill, 0.0)
            gap_size = max(0.0, required_level - current_level)
            
            if gap_size > 0:
                priority = self._calculate_priority(gap_size, skill, target_role)
                learning_path = self._generate_learning_path(skill, gap_size)
                
                gaps.append(SkillGap(
                    skill=skill,
                    current_level=current_level,
                    required_level=required_level,
                    gap_size=gap_size,
                    priority=priority,
                    learning_path=learning_path,
                    estimated_time=self._estimate_learning_time(gap_size),
                    resources=self._get_learning_resources(skill),
                ))
        
        # Ordenar por prioridad
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: priority_order.get(g.priority, 4))
        
        # Calcular scores
        overall_gap_score = self._calculate_overall_gap_score(gaps)
        readiness_score = 1.0 - overall_gap_score
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(gaps, readiness_score)
        
        # Crear plan de aprendizaje
        learning_plan = self._create_learning_plan(gaps)
        
        analysis = SkillGapAnalysis(
            user_id=user_id,
            target_role=target_role,
            current_skills=current_skills,
            required_skills=required_skills,
            gaps=gaps,
            overall_gap_score=overall_gap_score,
            readiness_score=readiness_score,
            recommendations=recommendations,
            learning_plan=learning_plan,
        )
        
        self.analyses[user_id] = analysis
        
        logger.info(f"Skill gap analysis completed for user {user_id}")
        return analysis
    
    def _get_required_skills_for_role(self, role: str) -> Dict[str, float]:
        """Obtener habilidades requeridas para un rol"""
        # En producción, esto consultaría una base de datos o API
        role_skills = {
            "Software Engineer": {
                "Programming": 0.8,
                "Algorithms": 0.7,
                "System Design": 0.6,
                "Testing": 0.7,
                "Version Control": 0.8,
            },
            "Data Scientist": {
                "Python": 0.9,
                "Machine Learning": 0.8,
                "Statistics": 0.8,
                "Data Visualization": 0.7,
                "SQL": 0.8,
            },
            "Product Manager": {
                "Product Strategy": 0.8,
                "User Research": 0.7,
                "Agile": 0.8,
                "Analytics": 0.7,
                "Communication": 0.9,
            },
        }
        
        return role_skills.get(role, {
            "Technical Skills": 0.7,
            "Communication": 0.8,
            "Problem Solving": 0.8,
        })
    
    def _calculate_priority(
        self,
        gap_size: float,
        skill: str,
        target_role: str
    ) -> str:
        """Calcular prioridad de la brecha"""
        if gap_size > 0.6:
            return "critical"
        elif gap_size > 0.4:
            return "high"
        elif gap_size > 0.2:
            return "medium"
        else:
            return "low"
    
    def _generate_learning_path(self, skill: str, gap_size: float) -> List[str]:
        """Generar ruta de aprendizaje"""
        paths = {
            "Programming": [
                "Completa curso básico de programación",
                "Practica con proyectos pequeños",
                "Contribuye a proyectos open source",
            ],
            "Machine Learning": [
                "Aprende fundamentos de ML",
                "Completa proyectos prácticos",
                "Lee papers y artículos recientes",
            ],
            "System Design": [
                "Estudia arquitecturas de sistemas",
                "Practica diseñando sistemas",
                "Revisa casos de estudio reales",
            ],
        }
        
        return paths.get(skill, [
            f"Investiga sobre {skill}",
            f"Practica {skill}",
            f"Aplica {skill} en proyectos",
        ])
    
    def _estimate_learning_time(self, gap_size: float) -> int:
        """Estimar tiempo de aprendizaje en días"""
        # Estimación: gap_size * 30 días base
        return int(gap_size * 30)
    
    def _get_learning_resources(self, skill: str) -> List[Dict[str, str]]:
        """Obtener recursos de aprendizaje"""
        return [
            {"type": "course", "name": f"{skill} Course", "url": "https://example.com"},
            {"type": "book", "name": f"{skill} Book", "url": "https://example.com"},
            {"type": "tutorial", "name": f"{skill} Tutorial", "url": "https://example.com"},
        ]
    
    def _calculate_overall_gap_score(self, gaps: List[SkillGap]) -> float:
        """Calcular score general de brechas"""
        if not gaps:
            return 0.0
        
        # Promedio ponderado por prioridad
        weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}
        weighted_sum = sum(g.gap_size * weights.get(g.priority, 0.2) for g in gaps)
        total_weight = sum(weights.get(g.priority, 0.2) for g in gaps)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(
        self,
        gaps: List[SkillGap],
        readiness_score: float
    ) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        critical_gaps = [g for g in gaps if g.priority == "critical"]
        if critical_gaps:
            recommendations.append(
                f"Enfócate en {len(critical_gaps)} habilidades críticas primero"
            )
        
        if readiness_score < 0.5:
            recommendations.append(
                "Considera roles intermedios antes del objetivo final"
            )
        elif readiness_score < 0.7:
            recommendations.append(
                "Estás cerca, enfócate en cerrar las brechas restantes"
            )
        else:
            recommendations.append(
                "Estás bien preparado, considera aplicar a posiciones"
            )
        
        # Recomendación de tiempo
        if gaps:
            max_time = max(g.estimated_time for g in gaps)
            recommendations.append(
                f"Planifica {max_time} días para cerrar todas las brechas"
            )
        
        return recommendations
    
    def _create_learning_plan(self, gaps: List[SkillGap]) -> Dict[str, Any]:
        """Crear plan de aprendizaje estructurado"""
        plan = {
            "total_skills": len(gaps),
            "critical_skills": len([g for g in gaps if g.priority == "critical"]),
            "estimated_total_time": sum(g.estimated_time for g in gaps),
            "phases": [],
        }
        
        # Organizar por prioridad
        for priority in ["critical", "high", "medium", "low"]:
            priority_gaps = [g for g in gaps if g.priority == priority]
            if priority_gaps:
                plan["phases"].append({
                    "phase": priority,
                    "skills": [g.skill for g in priority_gaps],
                    "estimated_time": sum(g.estimated_time for g in priority_gaps),
                })
        
        return plan
    
    def track_progress(
        self,
        user_id: str,
        skill: str,
        new_level: float
    ) -> Dict[str, Any]:
        """Rastrear progreso en una habilidad"""
        analysis = self.analyses.get(user_id)
        if not analysis:
            raise ValueError(f"Analysis not found for user {user_id}")
        
        # Actualizar nivel actual
        analysis.current_skills[skill] = new_level
        
        # Recalcular brecha
        for gap in analysis.gaps:
            if gap.skill == skill:
                gap.current_level = new_level
                gap.gap_size = max(0.0, gap.required_level - new_level)
                break
        
        # Recalcular scores
        analysis.overall_gap_score = self._calculate_overall_gap_score(analysis.gaps)
        analysis.readiness_score = 1.0 - analysis.overall_gap_score
        
        return {
            "skill": skill,
            "new_level": new_level,
            "updated_gap": gap.gap_size if gap else None,
            "readiness_score": analysis.readiness_score,
        }




