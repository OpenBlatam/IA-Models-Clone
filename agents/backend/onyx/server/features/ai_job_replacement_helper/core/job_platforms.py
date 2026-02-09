"""
Job Platforms Integration - Integración con múltiples plataformas
==================================================================

Integración con diversas plataformas de trabajo: Indeed, Glassdoor,
LinkedIn, y más.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Tipos de plataformas"""
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    MONSTER = "monster"
    ZIPRECRUITER = "ziprecruiter"
    REMOTE = "remote"
    STACKOVERFLOW = "stackoverflow"


@dataclass
class JobListing:
    """Listado de trabajo unificado"""
    id: str
    platform: PlatformType
    title: str
    company: str
    location: str
    description: str
    salary_range: Optional[str] = None
    job_type: Optional[str] = None
    posted_date: Optional[datetime] = None
    application_url: str = ""
    company_logo: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)
    match_score: float = 0.0
    platform_specific_data: Dict[str, Any] = field(default_factory=dict)


class JobPlatformsService:
    """Servicio de integración con plataformas"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.platform_configs: Dict[PlatformType, Dict[str, Any]] = {}
        logger.info("JobPlatformsService initialized")
    
    def search_jobs_across_platforms(
        self,
        user_id: str,
        keywords: str,
        location: Optional[str] = None,
        platforms: Optional[List[PlatformType]] = None,
        limit_per_platform: int = 10
    ) -> List[JobListing]:
        """Buscar trabajos en múltiples plataformas"""
        if not platforms:
            platforms = [
                PlatformType.LINKEDIN,
                PlatformType.INDEED,
                PlatformType.GLASSDOOR,
            ]
        
        all_jobs = []
        
        for platform in platforms:
            try:
                jobs = self._search_platform(
                    platform=platform,
                    keywords=keywords,
                    location=location,
                    limit=limit_per_platform
                )
                all_jobs.extend(jobs)
            except Exception as e:
                logger.error(f"Error searching {platform.value}: {e}")
        
        # Eliminar duplicados (mismo título y empresa)
        unique_jobs = self._deduplicate_jobs(all_jobs)
        
        # Ordenar por match score
        unique_jobs.sort(key=lambda x: x.match_score, reverse=True)
        
        return unique_jobs
    
    def _search_platform(
        self,
        platform: PlatformType,
        keywords: str,
        location: Optional[str],
        limit: int
    ) -> List[JobListing]:
        """Buscar en una plataforma específica"""
        if platform == PlatformType.LINKEDIN:
            return self._search_linkedin(keywords, location, limit)
        elif platform == PlatformType.INDEED:
            return self._search_indeed(keywords, location, limit)
        elif platform == PlatformType.GLASSDOOR:
            return self._search_glassdoor(keywords, location, limit)
        elif platform == PlatformType.REMOTE:
            return self._search_remote(keywords, limit)
        else:
            return []
    
    def _search_linkedin(
        self,
        keywords: str,
        location: Optional[str],
        limit: int
    ) -> List[JobListing]:
        """Buscar en LinkedIn (simulado)"""
        jobs = []
        for i in range(limit):
            job = JobListing(
                id=f"linkedin_{i+1}",
                platform=PlatformType.LINKEDIN,
                title=f"{keywords} Developer",
                company=f"Tech Company {i+1}",
                location=location or "Remote",
                description=f"Looking for a {keywords} developer...",
                salary_range=f"€{40000 + i*5000}-€{60000 + i*5000}",
                job_type="full-time",
                posted_date=datetime.now(),
                application_url=f"https://linkedin.com/jobs/view/{i+1}",
                required_skills=[keywords, "Python", "SQL"],
                match_score=0.7 + (i % 3) * 0.1,
            )
            jobs.append(job)
        return jobs
    
    def _search_indeed(
        self,
        keywords: str,
        location: Optional[str],
        limit: int
    ) -> List[JobListing]:
        """Buscar en Indeed (simulado)"""
        jobs = []
        for i in range(limit):
            job = JobListing(
                id=f"indeed_{i+1}",
                platform=PlatformType.INDEED,
                title=f"{keywords} Specialist",
                company=f"Company {i+1}",
                location=location or "Multiple Locations",
                description=f"Indeed job posting for {keywords}...",
                salary_range=f"€{35000 + i*4000}-€{55000 + i*4000}",
                job_type="full-time",
                posted_date=datetime.now(),
                application_url=f"https://indeed.com/viewjob?jk={i+1}",
                required_skills=[keywords, "Communication"],
                match_score=0.6 + (i % 3) * 0.1,
            )
            jobs.append(job)
        return jobs
    
    def _search_glassdoor(
        self,
        keywords: str,
        location: Optional[str],
        limit: int
    ) -> List[JobListing]:
        """Buscar en Glassdoor (simulado)"""
        jobs = []
        for i in range(limit):
            job = JobListing(
                id=f"glassdoor_{i+1}",
                platform=PlatformType.GLASSDOOR,
                title=f"{keywords} Engineer",
                company=f"Enterprise {i+1}",
                location=location or "Hybrid",
                description=f"Glassdoor listing for {keywords} engineer...",
                salary_range=f"€{45000 + i*6000}-€{70000 + i*6000}",
                job_type="full-time",
                posted_date=datetime.now(),
                application_url=f"https://glassdoor.com/job-listing/{i+1}",
                required_skills=[keywords, "Problem Solving"],
                match_score=0.65 + (i % 3) * 0.1,
            )
            jobs.append(job)
        return jobs
    
    def _search_remote(
        self,
        keywords: str,
        limit: int
    ) -> List[JobListing]:
        """Buscar trabajos remotos (simulado)"""
        jobs = []
        for i in range(limit):
            job = JobListing(
                id=f"remote_{i+1}",
                platform=PlatformType.REMOTE,
                title=f"Remote {keywords} Developer",
                company=f"Remote Company {i+1}",
                location="Remote",
                description=f"100% remote position for {keywords}...",
                salary_range=f"€{50000 + i*7000}-€{80000 + i*7000}",
                job_type="full-time",
                posted_date=datetime.now(),
                application_url=f"https://remote.com/jobs/{i+1}",
                required_skills=[keywords, "Remote Work"],
                match_score=0.75 + (i % 3) * 0.1,
            )
            jobs.append(job)
        return jobs
    
    def _deduplicate_jobs(self, jobs: List[JobListing]) -> List[JobListing]:
        """Eliminar trabajos duplicados"""
        seen = set()
        unique = []
        
        for job in jobs:
            # Crear clave única basada en título y empresa
            key = (job.title.lower().strip(), job.company.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(job)
        
        return unique
    
    def get_platform_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de plataformas"""
        return {
            "linkedin": {
                "name": "LinkedIn",
                "jobs_available": 1000000,
                "popularity": "high",
                "features": ["Networking", "Company Insights", "Easy Apply"],
            },
            "indeed": {
                "name": "Indeed",
                "jobs_available": 500000,
                "popularity": "high",
                "features": ["Wide Coverage", "Salary Info", "Company Reviews"],
            },
            "glassdoor": {
                "name": "Glassdoor",
                "jobs_available": 200000,
                "popularity": "medium",
                "features": ["Company Reviews", "Salary Data", "Interview Insights"],
            },
            "remote": {
                "name": "Remote.com",
                "jobs_available": 50000,
                "popularity": "medium",
                "features": ["100% Remote", "Global Opportunities"],
            },
        }




