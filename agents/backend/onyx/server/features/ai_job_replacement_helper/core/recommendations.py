"""
Recommendation Service - Sistema de recomendaciones inteligentes
==================================================================

Sistema de recomendaciones basado en IA para sugerir trabajos,
habilidades y pasos personalizados.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SkillRecommendation:
    """Recomendación de habilidad"""
    skill: str
    category: str
    priority: int  # 1-10
    reason: str
    learning_resources: List[Dict[str, str]]
    estimated_time: str
    market_demand: str  # high, medium, low


@dataclass
class JobRecommendation:
    """Recomendación de trabajo"""
    job_id: str
    title: str
    company: str
    match_score: float
    match_reasons: List[str]
    required_skills: List[str]
    missing_skills: List[str]
    skill_gap_score: float  # 0.0 a 1.0


class RecommendationService:
    """Servicio de recomendaciones inteligentes"""
    
    def __init__(self):
        """Inicializar servicio de recomendaciones"""
        # Habilidades en demanda (simulado)
        self.high_demand_skills = {
            "Python": {"category": "Programming", "demand": "high"},
            "JavaScript": {"category": "Programming", "demand": "high"},
            "Machine Learning": {"category": "AI/ML", "demand": "high"},
            "Data Analysis": {"category": "Data", "demand": "high"},
            "Cloud Computing": {"category": "Infrastructure", "demand": "high"},
            "DevOps": {"category": "Infrastructure", "demand": "high"},
            "UI/UX Design": {"category": "Design", "demand": "medium"},
            "Project Management": {"category": "Management", "demand": "medium"},
            "Digital Marketing": {"category": "Marketing", "demand": "medium"},
            "Content Creation": {"category": "Content", "demand": "medium"},
        }
        
        logger.info("RecommendationService initialized")
    
    def recommend_skills(
        self,
        user_skills: List[str],
        user_interests: List[str],
        target_industry: Optional[str] = None
    ) -> List[SkillRecommendation]:
        """Recomendar habilidades basadas en perfil del usuario"""
        recommendations = []
        
        # Analizar gap de habilidades
        missing_skills = self._identify_skill_gaps(user_skills, target_industry)
        
        for skill, info in missing_skills.items():
            if skill in self.high_demand_skills:
                skill_info = self.high_demand_skills[skill]
                
                # Calcular prioridad basada en demanda y relevancia
                priority = self._calculate_skill_priority(
                    skill,
                    user_skills,
                    user_interests,
                    skill_info["demand"]
                )
                
                recommendation = SkillRecommendation(
                    skill=skill,
                    category=skill_info["category"],
                    priority=priority,
                    reason=self._generate_skill_reason(skill, user_skills, skill_info),
                    learning_resources=self._get_learning_resources(skill),
                    estimated_time=self._estimate_learning_time(skill),
                    market_demand=skill_info["demand"]
                )
                recommendations.append(recommendation)
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations[:10]  # Top 10
    
    def recommend_jobs(
        self,
        user_skills: List[str],
        user_experience: Optional[str] = None,
        location: Optional[str] = None,
        job_preferences: Optional[Dict[str, Any]] = None
    ) -> List[JobRecommendation]:
        """Recomendar trabajos basados en perfil del usuario"""
        # En producción, esto usaría un modelo de ML real
        # Por ahora, simulamos recomendaciones
        
        recommendations = []
        
        # Trabajos simulados
        potential_jobs = [
            {
                "id": "job_1",
                "title": "Python Developer",
                "company": "TechCorp",
                "required_skills": ["Python", "FastAPI", "PostgreSQL"],
            },
            {
                "id": "job_2",
                "title": "Data Scientist",
                "company": "DataSolutions",
                "required_skills": ["Python", "Machine Learning", "SQL"],
            },
            {
                "id": "job_3",
                "title": "Full Stack Developer",
                "company": "StartupHub",
                "required_skills": ["JavaScript", "React", "Node.js"],
            },
        ]
        
        for job in potential_jobs:
            match_score, match_reasons = self._calculate_job_match(
                user_skills,
                job["required_skills"]
            )
            
            missing_skills = [
                skill for skill in job["required_skills"]
                if skill not in user_skills
            ]
            
            skill_gap_score = 1.0 - (len(missing_skills) / len(job["required_skills"]))
            
            recommendation = JobRecommendation(
                job_id=job["id"],
                title=job["title"],
                company=job["company"],
                match_score=match_score,
                match_reasons=match_reasons,
                required_skills=job["required_skills"],
                missing_skills=missing_skills,
                skill_gap_score=skill_gap_score
            )
            
            recommendations.append(recommendation)
        
        # Ordenar por match score
        recommendations.sort(key=lambda x: x.match_score, reverse=True)
        
        return recommendations
    
    def recommend_next_steps(
        self,
        completed_steps: List[str],
        user_skills: List[str],
        user_goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Recomendar próximos pasos personalizados"""
        recommendations = []
        
        # Si no ha completado evaluación inicial
        if "step_1" not in completed_steps:
            recommendations.append({
                "step_id": "step_1",
                "title": "Evalúa tu situación actual",
                "priority": "high",
                "reason": "Es el primer paso fundamental para crear tu plan",
            })
            return recommendations
        
        # Si no ha identificado habilidades
        if "step_2" not in completed_steps:
            recommendations.append({
                "step_id": "step_2",
                "title": "Identifica nuevas habilidades demandadas",
                "priority": "high",
                "reason": "Necesitas saber qué habilidades aprender",
            })
            return recommendations
        
        # Si tiene pocas habilidades, recomendar aprendizaje
        if len(user_skills) < 3:
            recommendations.append({
                "step_id": "step_3",
                "title": "Crea un plan de aprendizaje",
                "priority": "high",
                "reason": "Tienes pocas habilidades, enfócate en aprender primero",
            })
        
        # Si tiene habilidades pero no ha actualizado LinkedIn
        if len(user_skills) >= 3 and "step_4" not in completed_steps:
            recommendations.append({
                "step_id": "step_4",
                "title": "Actualiza tu perfil de LinkedIn",
                "priority": "medium",
                "reason": "Ya tienes habilidades, es hora de mostrarlas",
            })
        
        return recommendations
    
    def _identify_skill_gaps(
        self,
        user_skills: List[str],
        target_industry: Optional[str]
    ) -> Dict[str, Dict[str, str]]:
        """Identificar gaps de habilidades"""
        gaps = {}
        
        for skill, info in self.high_demand_skills.items():
            if skill not in user_skills:
                gaps[skill] = info
        
        return gaps
    
    def _calculate_skill_priority(
        self,
        skill: str,
        user_skills: List[str],
        user_interests: List[str],
        demand: str
    ) -> int:
        """Calcular prioridad de una habilidad (1-10)"""
        priority = 5  # Base
        
        # Aumentar si está en alta demanda
        if demand == "high":
            priority += 2
        elif demand == "medium":
            priority += 1
        
        # Aumentar si está relacionada con habilidades existentes
        related_skills = {
            "Python": ["Programming", "Data Analysis"],
            "JavaScript": ["Programming", "Web Development"],
            "Machine Learning": ["Python", "Data Analysis"],
        }
        
        if skill in related_skills:
            for related in related_skills[skill]:
                if related in user_skills:
                    priority += 1
                    break
        
        # Aumentar si está en intereses del usuario
        if any(interest.lower() in skill.lower() for interest in user_interests):
            priority += 1
        
        return min(priority, 10)
    
    def _generate_skill_reason(
        self,
        skill: str,
        user_skills: List[str],
        skill_info: Dict[str, str]
    ) -> str:
        """Generar razón para recomendar una habilidad"""
        reasons = [
            f"{skill} está en alta demanda en el mercado actual",
            f"Completa tu perfil de {skill_info['category']}",
            f"Se complementa bien con tus habilidades actuales",
        ]
        
        if user_skills:
            reasons.append(f"Te permitirá expandir tus oportunidades más allá de {user_skills[0]}")
        
        return reasons[0]  # Por ahora, devolver la primera
    
    def _get_learning_resources(self, skill: str) -> List[Dict[str, str]]:
        """Obtener recursos de aprendizaje para una habilidad"""
        return [
            {
                "type": "course",
                "title": f"Curso completo de {skill}",
                "url": f"#course_{skill.lower()}",
                "platform": "Coursera",
            },
            {
                "type": "tutorial",
                "title": f"Tutorial práctico de {skill}",
                "url": f"#tutorial_{skill.lower()}",
                "platform": "YouTube",
            },
            {
                "type": "documentation",
                "title": f"Documentación oficial de {skill}",
                "url": f"#docs_{skill.lower()}",
                "platform": "Official",
            },
        ]
    
    def _estimate_learning_time(self, skill: str) -> str:
        """Estimar tiempo de aprendizaje"""
        estimates = {
            "Python": "2-3 meses",
            "JavaScript": "2-3 meses",
            "Machine Learning": "4-6 meses",
            "Data Analysis": "2-3 meses",
            "Cloud Computing": "3-4 meses",
            "DevOps": "3-4 meses",
        }
        
        return estimates.get(skill, "2-4 meses")
    
    def _calculate_job_match(
        self,
        user_skills: List[str],
        required_skills: List[str]
    ) -> Tuple[float, List[str]]:
        """Calcular match score entre usuario y trabajo"""
        if not required_skills:
            return 0.0, []
        
        # Calcular porcentaje de habilidades que tiene el usuario
        matching_skills = [s for s in required_skills if s in user_skills]
        match_ratio = len(matching_skills) / len(required_skills)
        
        match_score = match_ratio
        
        # Generar razones
        reasons = []
        if match_ratio >= 0.8:
            reasons.append("Tienes la mayoría de las habilidades requeridas")
        elif match_ratio >= 0.5:
            reasons.append("Tienes algunas de las habilidades requeridas")
        else:
            reasons.append("Necesitas aprender más habilidades para este trabajo")
        
        if matching_skills:
            reasons.append(f"Ya tienes: {', '.join(matching_skills[:3])}")
        
        return match_score, reasons

