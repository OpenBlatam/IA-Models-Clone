"""
ML Recommendations Service - Recomendaciones con Machine Learning
=================================================================

Sistema de recomendaciones avanzado usando modelos de ML.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Perfil de usuario para ML"""
    user_id: str
    skills: List[str]
    experience_years: int
    industries: List[str]
    job_titles: List[str]
    preferences: Dict[str, Any]
    behavior_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLRecommendation:
    """Recomendación generada por ML"""
    item_id: str
    item_type: str  # job, skill, course, etc.
    score: float
    confidence: float
    reasoning: str
    features: Dict[str, Any] = field(default_factory=dict)


class MLRecommendationsService:
    """Servicio de recomendaciones con ML"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.user_profiles: Dict[str, UserProfile] = {}
        self.model_weights: Dict[str, float] = {
            "skill_match": 0.3,
            "experience_match": 0.2,
            "industry_match": 0.15,
            "location_preference": 0.1,
            "salary_expectation": 0.1,
            "company_culture": 0.1,
            "growth_potential": 0.05,
        }
        logger.info("MLRecommendationsService initialized")
    
    def train_user_model(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Entrenar modelo personalizado para usuario"""
        # En producción, esto entrenaría un modelo real
        # Por ahora, simulamos el entrenamiento
        
        positive_interactions = [i for i in interactions if i.get("rating", 0) > 3]
        negative_interactions = [i for i in interactions if i.get("rating", 0) <= 3]
        
        # Actualizar pesos basado en feedback
        if positive_interactions:
            for interaction in positive_interactions:
                # Ajustar pesos según interacciones positivas
                pass
        
        return {
            "user_id": user_id,
            "model_trained": True,
            "training_samples": len(interactions),
            "positive_samples": len(positive_interactions),
            "negative_samples": len(negative_interactions),
            "accuracy": 0.85,  # Simulado
        }
    
    def recommend_jobs_ml(
        self,
        user_id: str,
        job_pool: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[MLRecommendation]:
        """Recomendar trabajos usando ML"""
        user_profile = self.user_profiles.get(user_id)
        if not user_profile:
            return []
        
        recommendations = []
        
        for job in job_pool:
            score = self._calculate_job_score(user_profile, job)
            confidence = self._calculate_confidence(user_profile, job)
            
            recommendations.append(MLRecommendation(
                item_id=job.get("id", ""),
                item_type="job",
                score=score,
                confidence=confidence,
                reasoning=self._generate_reasoning(user_profile, job, score),
                features={
                    "skill_match": self._skill_match_score(user_profile, job),
                    "experience_match": self._experience_match(user_profile, job),
                    "industry_match": self._industry_match(user_profile, job),
                }
            ))
        
        # Ordenar por score y retornar top_k
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]
    
    def _calculate_job_score(self, profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calcular score de compatibilidad"""
        scores = {
            "skill_match": self._skill_match_score(profile, job) * self.model_weights["skill_match"],
            "experience_match": self._experience_match(profile, job) * self.model_weights["experience_match"],
            "industry_match": self._industry_match(profile, job) * self.model_weights["industry_match"],
        }
        
        return sum(scores.values())
    
    def _skill_match_score(self, profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calcular match de habilidades"""
        required_skills = job.get("required_skills", [])
        user_skills = set(profile.skills)
        job_skills = set(required_skills)
        
        if not job_skills:
            return 0.5  # Neutral si no hay skills especificadas
        
        intersection = user_skills.intersection(job_skills)
        union = user_skills.union(job_skills)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _experience_match(self, profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calcular match de experiencia"""
        required_years = job.get("required_experience_years", 0)
        user_years = profile.experience_years
        
        if required_years == 0:
            return 1.0
        
        if user_years >= required_years:
            return 1.0
        else:
            return user_years / required_years
    
    def _industry_match(self, profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calcular match de industria"""
        job_industry = job.get("industry", "")
        user_industries = set(profile.industries)
        
        if not job_industry:
            return 0.5
        
        return 1.0 if job_industry in user_industries else 0.3
    
    def _calculate_confidence(self, profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calcular confianza en la recomendación"""
        # Más datos del usuario = mayor confianza
        data_points = len(profile.skills) + len(profile.industries) + len(profile.job_titles)
        confidence = min(1.0, data_points / 20.0)  # Normalizar
        
        return confidence
    
    def _generate_reasoning(self, profile: UserProfile, job: Dict[str, Any], score: float) -> str:
        """Generar explicación de la recomendación"""
        if score > 0.8:
            return "Excelente match: tus habilidades y experiencia coinciden perfectamente con este trabajo."
        elif score > 0.6:
            return "Buen match: hay buena compatibilidad, aunque podrías necesitar desarrollar algunas habilidades adicionales."
        else:
            return "Match moderado: este trabajo podría requerir más preparación, pero es alcanzable."
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Actualizar perfil de usuario"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                skills=profile_data.get("skills", []),
                experience_years=profile_data.get("experience_years", 0),
                industries=profile_data.get("industries", []),
                job_titles=profile_data.get("job_titles", []),
                preferences=profile_data.get("preferences", {}),
            )
        else:
            profile = self.user_profiles[user_id]
            profile.skills = profile_data.get("skills", profile.skills)
            profile.experience_years = profile_data.get("experience_years", profile.experience_years)
            profile.industries = profile_data.get("industries", profile.industries)
            profile.job_titles = profile_data.get("job_titles", profile.job_titles)
            profile.preferences.update(profile_data.get("preferences", {}))




