"""
AI Resume Builder Service - Constructor de CV con IA
======================================================

Sistema para construir y optimizar CVs usando IA.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ResumeFormat(str, Enum):
    """Formatos de CV"""
    CHRONOLOGICAL = "chronological"
    FUNCTIONAL = "functional"
    COMBINATION = "combination"
    ATS_FRIENDLY = "ats_friendly"


@dataclass
class ResumeSection:
    """Sección del CV"""
    section_type: str  # "header", "experience", "education", "skills", etc.
    content: Dict[str, Any]
    order: int


@dataclass
class Resume:
    """CV completo"""
    id: str
    user_id: str
    title: str
    format: ResumeFormat
    sections: List[ResumeSection]
    target_job: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    ats_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class AIResumeBuilderService:
    """Servicio de construcción de CV con IA"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.resumes: Dict[str, Resume] = {}
        logger.info("AIResumeBuilderService initialized")
    
    def create_resume(
        self,
        user_id: str,
        title: str,
        format: ResumeFormat = ResumeFormat.ATS_FRIENDLY,
        target_job: Optional[str] = None
    ) -> Resume:
        """Crear nuevo CV"""
        resume_id = f"resume_{user_id}_{int(datetime.now().timestamp())}"
        
        resume = Resume(
            id=resume_id,
            user_id=user_id,
            title=title,
            format=format,
            sections=[],
            target_job=target_job,
        )
        
        self.resumes[resume_id] = resume
        
        logger.info(f"Resume created: {resume_id}")
        return resume
    
    def add_section(
        self,
        resume_id: str,
        section_type: str,
        content: Dict[str, Any],
        order: Optional[int] = None
    ) -> ResumeSection:
        """Agregar sección al CV"""
        resume = self.resumes.get(resume_id)
        if not resume:
            raise ValueError(f"Resume {resume_id} not found")
        
        if order is None:
            order = len(resume.sections)
        
        section = ResumeSection(
            section_type=section_type,
            content=content,
            order=order,
        )
        
        resume.sections.append(section)
        resume.sections.sort(key=lambda s: s.order)
        resume.updated_at = datetime.now()
        
        return section
    
    def optimize_for_job(
        self,
        resume_id: str,
        job_description: str
    ) -> Dict[str, Any]:
        """Optimizar CV para un trabajo específico"""
        resume = self.resumes.get(resume_id)
        if not resume:
            raise ValueError(f"Resume {resume_id} not found")
        
        # Extraer keywords del job description
        keywords = self._extract_keywords(job_description)
        resume.keywords = keywords
        
        # Optimizar contenido
        optimized_sections = []
        for section in resume.sections:
            optimized_content = self._optimize_section(section, keywords)
            optimized_sections.append({
                "section_type": section.section_type,
                "original": section.content,
                "optimized": optimized_content,
                "improvements": self._suggest_improvements(section, keywords),
            })
        
        # Calcular ATS score
        ats_score = self._calculate_ats_score(resume, keywords)
        resume.ats_score = ats_score
        
        return {
            "resume_id": resume_id,
            "keywords_found": keywords,
            "optimized_sections": optimized_sections,
            "ats_score": ats_score,
            "recommendations": self._generate_recommendations(resume, ats_score),
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extraer keywords del texto"""
        # En producción, esto usaría NLP/IA real
        # Por ahora, simulamos
        common_keywords = [
            "Python", "JavaScript", "React", "AWS", "Docker",
            "Agile", "Scrum", "Leadership", "Communication",
            "Machine Learning", "Data Science", "SQL",
        ]
        
        found_keywords = [kw for kw in common_keywords if kw.lower() in text.lower()]
        return found_keywords[:10]  # Top 10
    
    def _optimize_section(
        self,
        section: ResumeSection,
        keywords: List[str]
    ) -> Dict[str, Any]:
        """Optimizar sección con keywords"""
        content = section.content.copy()
        
        # Agregar keywords relevantes si no están presentes
        if section.section_type == "skills":
            existing_skills = content.get("skills", [])
            missing_keywords = [kw for kw in keywords if kw not in existing_skills]
            if missing_keywords:
                content["suggested_additions"] = missing_keywords[:3]
        
        return content
    
    def _suggest_improvements(
        self,
        section: ResumeSection,
        keywords: List[str]
    ) -> List[str]:
        """Sugerir mejoras para la sección"""
        improvements = []
        
        if section.section_type == "experience":
            improvements.append("Usa números y métricas para cuantificar logros")
            improvements.append("Incluye keywords relevantes del job description")
        
        if section.section_type == "skills":
            missing = [kw for kw in keywords if kw.lower() not in str(section.content).lower()]
            if missing:
                improvements.append(f"Considera agregar: {', '.join(missing[:3])}")
        
        return improvements
    
    def _calculate_ats_score(
        self,
        resume: Resume,
        keywords: List[str]
    ) -> float:
        """Calcular score ATS"""
        if not keywords:
            return 0.5
        
        # Contar keywords presentes en el CV
        resume_text = str(resume.sections).lower()
        found_keywords = sum(1 for kw in keywords if kw.lower() in resume_text)
        
        # Score basado en porcentaje de keywords encontradas
        score = found_keywords / len(keywords)
        
        return round(score, 2)
    
    def _generate_recommendations(
        self,
        resume: Resume,
        ats_score: float
    ) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        if ats_score < 0.5:
            recommendations.append(
                "Agrega más keywords del job description a tu CV"
            )
            recommendations.append(
                "Asegúrate de que las habilidades requeridas estén presentes"
            )
        elif ats_score < 0.7:
            recommendations.append(
                "Buen match de keywords. Considera optimizar el orden de secciones"
            )
        else:
            recommendations.append(
                "Excelente match de keywords. Tu CV está bien optimizado para ATS"
            )
        
        if resume.format != ResumeFormat.ATS_FRIENDLY:
            recommendations.append(
                "Considera usar formato ATS-friendly para mejor compatibilidad"
            )
        
        return recommendations
    
    def generate_resume_pdf(self, resume_id: str) -> Dict[str, Any]:
        """Generar PDF del CV"""
        resume = self.resumes.get(resume_id)
        if not resume:
            raise ValueError(f"Resume {resume_id} not found")
        
        # En producción, esto generaría un PDF real
        # Por ahora, retornamos metadata
        return {
            "resume_id": resume_id,
            "title": resume.title,
            "format": resume.format.value,
            "sections_count": len(resume.sections),
            "ats_score": resume.ats_score,
            "pdf_url": f"/resumes/{resume_id}.pdf",  # Simulado
        }
    
    def get_resume_analysis(self, resume_id: str) -> Dict[str, Any]:
        """Obtener análisis completo del CV"""
        resume = self.resumes.get(resume_id)
        if not resume:
            raise ValueError(f"Resume {resume_id} not found")
        
        return {
            "resume_id": resume_id,
            "title": resume.title,
            "format": resume.format.value,
            "sections": [
                {
                    "type": s.section_type,
                    "order": s.order,
                    "content_preview": str(s.content)[:100],
                }
                for s in resume.sections
            ],
            "ats_score": resume.ats_score,
            "keywords": resume.keywords,
            "target_job": resume.target_job,
            "last_updated": resume.updated_at.isoformat(),
        }




