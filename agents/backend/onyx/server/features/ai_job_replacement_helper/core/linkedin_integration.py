"""
LinkedIn Integration Service - Integración estilo Tinder
=========================================================

Sistema de búsqueda de trabajo estilo Tinder usando LinkedIn API.
Permite hacer swipe (like/dislike) de trabajos y matching inteligente.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import httpx
import os

logger = logging.getLogger(__name__)


class JobAction(str, Enum):
    """Acciones sobre trabajos"""
    LIKE = "like"
    DISLIKE = "dislike"
    SAVE = "save"
    APPLY = "apply"
    SKIP = "skip"


@dataclass
class Job:
    """Representa un trabajo de LinkedIn"""
    id: str
    title: str
    company: str
    location: str
    description: str
    salary_range: Optional[str] = None
    job_type: Optional[str] = None  # full-time, part-time, contract, etc.
    posted_date: Optional[datetime] = None
    application_url: Optional[str] = None
    company_logo: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    match_score: float = 0.0  # 0.0 a 1.0
    match_reasons: List[str] = field(default_factory=list)


@dataclass
class UserJobInteraction:
    """Interacción del usuario con un trabajo"""
    user_id: str
    job_id: str
    action: JobAction
    timestamp: datetime
    notes: Optional[str] = None


class LinkedInIntegrationService:
    """Servicio de integración con LinkedIn"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Inicializar servicio de LinkedIn"""
        self.api_key = api_key or os.getenv("LINKEDIN_API_KEY")
        self.base_url = "https://api.linkedin.com/v2"
        self.user_interactions: Dict[str, List[UserJobInteraction]] = {}
        self.saved_jobs: Dict[str, List[str]] = {}  # user_id -> [job_ids]
        self.liked_jobs: Dict[str, List[str]] = {}  # user_id -> [job_ids]
        
        logger.info("LinkedInIntegrationService initialized")
    
    async def search_jobs(
        self,
        user_id: str,
        keywords: Optional[str] = None,
        location: Optional[str] = None,
        experience_level: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Job]:
        """
        Buscar trabajos en LinkedIn
        
        Nota: Esta es una implementación simulada.
        En producción, usarías la LinkedIn Jobs API real.
        """
        logger.info(f"Searching jobs for user {user_id}")
        
        # Simulación de búsqueda de trabajos
        # En producción, esto haría una llamada real a la API de LinkedIn
        mock_jobs = self._generate_mock_jobs(keywords, location, limit)
        
        # Filtrar trabajos ya vistos/interactuados
        seen_jobs = self._get_seen_jobs(user_id)
        new_jobs = [job for job in mock_jobs if job.id not in seen_jobs]
        
        return new_jobs[:limit]
    
    def _generate_mock_jobs(
        self,
        keywords: Optional[str],
        location: Optional[str],
        count: int
    ) -> List[Job]:
        """Generar trabajos mock para desarrollo/testing"""
        # En producción, esto vendría de la API real de LinkedIn
        jobs = []
        
        job_titles = [
            "Desarrollador Python Senior",
            "Data Scientist",
            "Machine Learning Engineer",
            "Full Stack Developer",
            "DevOps Engineer",
            "Product Manager",
            "UX/UI Designer",
            "Marketing Digital Specialist",
            "Sales Representative",
            "Content Creator",
        ]
        
        companies = [
            "TechCorp",
            "DataSolutions",
            "AI Innovations",
            "CloudServices",
            "Digital Agency",
            "StartupHub",
            "Enterprise Systems",
            "Creative Studio",
        ]
        
        locations = [
            "Madrid, España",
            "Barcelona, España",
            "Remote",
            "Valencia, España",
            "Sevilla, España",
        ]
        
        for i in range(count):
            job = Job(
                id=f"job_{i+1}",
                title=job_titles[i % len(job_titles)],
                company=companies[i % len(companies)],
                location=location or locations[i % len(locations)],
                description=f"Estamos buscando un {job_titles[i % len(job_titles)]} para unirse a nuestro equipo. "
                           f"Oportunidad emocionante en una empresa en crecimiento.",
                salary_range=f"€{30000 + i * 5000}-€{50000 + i * 5000}",
                job_type="full-time",
                posted_date=datetime.now(),
                application_url=f"https://linkedin.com/jobs/view/{i+1}",
                required_skills=["Python", "SQL", "Git"],
                preferred_skills=["Docker", "AWS", "FastAPI"],
                match_score=0.7 + (i % 3) * 0.1,
                match_reasons=[
                    "Tus habilidades coinciden con los requisitos",
                    "Ubicación compatible",
                    "Nivel de experiencia adecuado",
                ]
            )
            jobs.append(job)
        
        return jobs
    
    def swipe_job(
        self,
        user_id: str,
        job_id: str,
        action: JobAction
    ) -> Dict[str, Any]:
        """Hacer swipe (like/dislike) en un trabajo"""
        interaction = UserJobInteraction(
            user_id=user_id,
            job_id=job_id,
            action=action,
            timestamp=datetime.now()
        )
        
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        
        self.user_interactions[user_id].append(interaction)
        
        # Actualizar listas según acción
        if action == JobAction.LIKE:
            if user_id not in self.liked_jobs:
                self.liked_jobs[user_id] = []
            if job_id not in self.liked_jobs[user_id]:
                self.liked_jobs[user_id].append(job_id)
        
        if action == JobAction.SAVE:
            if user_id not in self.saved_jobs:
                self.saved_jobs[user_id] = []
            if job_id not in self.saved_jobs[user_id]:
                self.saved_jobs[user_id].append(job_id)
        
        logger.info(f"User {user_id} {action.value} job {job_id}")
        
        return {
            "success": True,
            "action": action.value,
            "job_id": job_id,
            "timestamp": interaction.timestamp.isoformat(),
        }
    
    def get_liked_jobs(self, user_id: str) -> List[str]:
        """Obtener trabajos que le gustaron al usuario"""
        return self.liked_jobs.get(user_id, [])
    
    def get_saved_jobs(self, user_id: str) -> List[str]:
        """Obtener trabajos guardados por el usuario"""
        return self.saved_jobs.get(user_id, [])
    
    def get_job_matches(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener trabajos con match (mutual like)"""
        # En producción, esto verificaría si la empresa también mostró interés
        liked = self.get_liked_jobs(user_id)
        
        matches = []
        for job_id in liked:
            # Simulación: algunos trabajos tienen match
            if hash(f"{user_id}_{job_id}") % 3 == 0:
                matches.append({
                    "job_id": job_id,
                    "matched_at": datetime.now().isoformat(),
                    "status": "pending",
                })
        
        return matches
    
    def apply_to_job(
        self,
        user_id: str,
        job_id: str,
        cover_letter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aplicar a un trabajo"""
        # Registrar aplicación
        self.swipe_job(user_id, job_id, JobAction.APPLY)
        
        logger.info(f"User {user_id} applied to job {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "application_id": f"app_{user_id}_{job_id}_{int(datetime.now().timestamp())}",
            "applied_at": datetime.now().isoformat(),
            "status": "submitted",
        }
    
    def _get_seen_jobs(self, user_id: str) -> set:
        """Obtener IDs de trabajos ya vistos por el usuario"""
        if user_id not in self.user_interactions:
            return set()
        
        return {
            interaction.job_id
            for interaction in self.user_interactions[user_id]
        }
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Obtener estadísticas del usuario"""
        interactions = self.user_interactions.get(user_id, [])
        
        return {
            "total_viewed": len(set(i.job_id for i in interactions)),
            "total_liked": len(self.get_liked_jobs(user_id)),
            "total_saved": len(self.get_saved_jobs(user_id)),
            "total_applied": sum(
                1 for i in interactions if i.action == JobAction.APPLY
            ),
            "matches": len(self.get_job_matches(user_id)),
        }




