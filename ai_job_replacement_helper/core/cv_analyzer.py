"""
CV Analyzer Service - Análisis de CV con IA
============================================

Sistema que analiza CVs y proporciona feedback para mejorarlos.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CVSection(str, Enum):
    """Secciones del CV"""
    HEADER = "header"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    LANGUAGES = "languages"
    CERTIFICATIONS = "certifications"


@dataclass
class CVFeedback:
    """Feedback sobre el CV"""
    section: CVSection
    score: float  # 0.0 a 1.0
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    priority: str  # high, medium, low


@dataclass
class CVAnalysis:
    """Análisis completo del CV"""
    cv_id: str
    user_id: str
    overall_score: float
    ats_score: float  # Score para sistemas ATS
    feedback_by_section: Dict[str, CVFeedback]
    missing_elements: List[str]
    keyword_analysis: Dict[str, Any]
    suggestions: List[str]
    analyzed_at: datetime = field(default_factory=datetime.now)


class CVAnalyzerService:
    """Servicio de análisis de CV"""
    
    def __init__(self):
        """Inicializar servicio de análisis"""
        self.analyses: Dict[str, CVAnalysis] = {}
        logger.info("CVAnalyzerService initialized")
    
    def analyze_cv(
        self,
        user_id: str,
        cv_content: Dict[str, Any],
        target_job: Optional[Dict[str, Any]] = None
    ) -> CVAnalysis:
        """
        Analizar un CV
        
        Args:
            user_id: ID del usuario
            cv_content: Contenido del CV en formato estructurado
            target_job: Información del trabajo objetivo (opcional)
        """
        cv_id = f"cv_{user_id}_{int(datetime.now().timestamp())}"
        
        # Analizar cada sección
        feedback_by_section = {}
        
        # Análisis de header
        if "header" in cv_content:
            feedback_by_section["header"] = self._analyze_header(cv_content["header"])
        
        # Análisis de summary
        if "summary" in cv_content:
            feedback_by_section["summary"] = self._analyze_summary(cv_content["summary"])
        
        # Análisis de experiencia
        if "experience" in cv_content:
            feedback_by_section["experience"] = self._analyze_experience(cv_content["experience"])
        
        # Análisis de habilidades
        if "skills" in cv_content:
            feedback_by_section["skills"] = self._analyze_skills(cv_content["skills"], target_job)
        
        # Análisis de educación
        if "education" in cv_content:
            feedback_by_section["education"] = self._analyze_education(cv_content["education"])
        
        # Calcular scores
        overall_score = self._calculate_overall_score(feedback_by_section)
        ats_score = self._calculate_ats_score(cv_content, target_job)
        
        # Identificar elementos faltantes
        missing_elements = self._identify_missing_elements(cv_content)
        
        # Análisis de keywords
        keyword_analysis = self._analyze_keywords(cv_content, target_job)
        
        # Sugerencias generales
        suggestions = self._generate_suggestions(feedback_by_section, missing_elements)
        
        analysis = CVAnalysis(
            cv_id=cv_id,
            user_id=user_id,
            overall_score=overall_score,
            ats_score=ats_score,
            feedback_by_section=feedback_by_section,
            missing_elements=missing_elements,
            keyword_analysis=keyword_analysis,
            suggestions=suggestions
        )
        
        self.analyses[cv_id] = analysis
        
        logger.info(f"CV analyzed for user {user_id}: overall_score={overall_score:.2f}")
        return analysis
    
    def _analyze_header(self, header: Dict[str, Any]) -> CVFeedback:
        """Analizar sección de header"""
        strengths = []
        weaknesses = []
        suggestions = []
        
        if header.get("name"):
            strengths.append("Nombre presente")
        else:
            weaknesses.append("Falta el nombre")
            suggestions.append("Agrega tu nombre completo")
        
        if header.get("email"):
            strengths.append("Email presente")
        else:
            weaknesses.append("Falta el email")
            suggestions.append("Agrega un email profesional")
        
        if header.get("phone"):
            strengths.append("Teléfono presente")
        else:
            weaknesses.append("Falta el teléfono")
            suggestions.append("Agrega tu número de teléfono")
        
        if header.get("linkedin"):
            strengths.append("LinkedIn presente")
        else:
            suggestions.append("Agrega tu perfil de LinkedIn")
        
        score = len(strengths) / max(len(strengths) + len(weaknesses), 1)
        
        return CVFeedback(
            section=CVSection.HEADER,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            priority="high" if score < 0.7 else "medium"
        )
    
    def _analyze_summary(self, summary: str) -> CVFeedback:
        """Analizar resumen profesional"""
        strengths = []
        weaknesses = []
        suggestions = []
        
        word_count = len(summary.split()) if summary else 0
        
        if word_count >= 50 and word_count <= 200:
            strengths.append("Longitud apropiada del resumen")
        elif word_count < 50:
            weaknesses.append("Resumen muy corto")
            suggestions.append("Amplía tu resumen a 50-200 palabras")
        else:
            weaknesses.append("Resumen muy largo")
            suggestions.append("Acorta tu resumen a máximo 200 palabras")
        
        if summary and any(keyword in summary.lower() for keyword in ["experiencia", "habilidades", "logros"]):
            strengths.append("Incluye palabras clave relevantes")
        else:
            suggestions.append("Incluye palabras clave como 'experiencia', 'habilidades', 'logros'")
        
        score = len(strengths) / max(len(strengths) + len(weaknesses), 1)
        
        return CVFeedback(
            section=CVSection.SUMMARY,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            priority="high"
        )
    
    def _analyze_experience(self, experience: List[Dict[str, Any]]) -> CVFeedback:
        """Analizar experiencia laboral"""
        strengths = []
        weaknesses = []
        suggestions = []
        
        if len(experience) > 0:
            strengths.append("Tiene experiencia laboral")
        else:
            weaknesses.append("No hay experiencia laboral")
            suggestions.append("Agrega proyectos personales o voluntariado si no tienes experiencia")
        
        # Verificar que cada experiencia tenga elementos clave
        for exp in experience:
            if not exp.get("title"):
                weaknesses.append("Falta título en alguna experiencia")
                suggestions.append("Asegúrate de incluir el título del puesto")
            
            if not exp.get("company"):
                weaknesses.append("Falta empresa en alguna experiencia")
                suggestions.append("Incluye el nombre de la empresa")
            
            if exp.get("achievements") and len(exp["achievements"]) > 0:
                strengths.append("Incluye logros cuantificables")
            else:
                suggestions.append("Agrega logros con números y métricas")
        
        score = len(strengths) / max(len(strengths) + len(weaknesses), 1) if (strengths or weaknesses) else 0.5
        
        return CVFeedback(
            section=CVSection.EXPERIENCE,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            priority="high"
        )
    
    def _analyze_skills(
        self,
        skills: List[str],
        target_job: Optional[Dict[str, Any]]
    ) -> CVFeedback:
        """Analizar habilidades"""
        strengths = []
        weaknesses = []
        suggestions = []
        
        if len(skills) >= 5:
            strengths.append("Tiene suficientes habilidades listadas")
        else:
            weaknesses.append("Pocas habilidades listadas")
            suggestions.append("Agrega más habilidades relevantes")
        
        # Comparar con trabajo objetivo
        if target_job and "required_skills" in target_job:
            required = set(target_job["required_skills"])
            user_skills = set(skills)
            missing = required - user_skills
            
            if missing:
                weaknesses.append(f"Faltan habilidades requeridas: {', '.join(list(missing)[:3])}")
                suggestions.append(f"Considera aprender: {', '.join(list(missing)[:3])}")
            else:
                strengths.append("Tienes todas las habilidades requeridas")
        
        score = len(strengths) / max(len(strengths) + len(weaknesses), 1) if (strengths or weaknesses) else 0.7
        
        return CVFeedback(
            section=CVSection.SKILLS,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            priority="high"
        )
    
    def _analyze_education(self, education: List[Dict[str, Any]]) -> CVFeedback:
        """Analizar educación"""
        strengths = []
        weaknesses = []
        suggestions = []
        
        if len(education) > 0:
            strengths.append("Tiene información educativa")
        else:
            weaknesses.append("Falta información educativa")
            suggestions.append("Agrega tu educación formal o cursos relevantes")
        
        score = 0.8 if strengths else 0.3
        
        return CVFeedback(
            section=CVSection.EDUCATION,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            priority="medium"
        )
    
    def _calculate_overall_score(self, feedback_by_section: Dict[str, CVFeedback]) -> float:
        """Calcular score general"""
        if not feedback_by_section:
            return 0.0
        
        scores = [feedback.score for feedback in feedback_by_section.values()]
        return sum(scores) / len(scores)
    
    def _calculate_ats_score(
        self,
        cv_content: Dict[str, Any],
        target_job: Optional[Dict[str, Any]]
    ) -> float:
        """Calcular score para sistemas ATS (Applicant Tracking Systems)"""
        score = 0.5  # Base
        
        # Verificar keywords
        if target_job and "required_skills" in target_job:
            cv_text = str(cv_content).lower()
            required_skills = [s.lower() for s in target_job["required_skills"]]
            matches = sum(1 for skill in required_skills if skill in cv_text)
            score += (matches / len(required_skills)) * 0.3
        
        # Verificar formato estructurado
        if "header" in cv_content and "experience" in cv_content:
            score += 0.2
        
        return min(score, 1.0)
    
    def _identify_missing_elements(self, cv_content: Dict[str, Any]) -> List[str]:
        """Identificar elementos faltantes"""
        missing = []
        
        required_sections = ["header", "summary", "experience", "skills"]
        for section in required_sections:
            if section not in cv_content or not cv_content[section]:
                missing.append(section)
        
        return missing
    
    def _analyze_keywords(
        self,
        cv_content: Dict[str, Any],
        target_job: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar keywords"""
        cv_text = str(cv_content).lower()
        
        analysis = {
            "total_keywords": 0,
            "matched_keywords": [],
            "missing_keywords": [],
        }
        
        if target_job and "required_skills" in target_job:
            required_skills = [s.lower() for s in target_job["required_skills"]]
            analysis["total_keywords"] = len(required_skills)
            
            for skill in required_skills:
                if skill in cv_text:
                    analysis["matched_keywords"].append(skill)
                else:
                    analysis["missing_keywords"].append(skill)
        
        return analysis
    
    def _generate_suggestions(
        self,
        feedback_by_section: Dict[str, CVFeedback],
        missing_elements: List[str]
    ) -> List[str]:
        """Generar sugerencias generales"""
        suggestions = []
        
        # Sugerencias basadas en secciones faltantes
        if "summary" in missing_elements:
            suggestions.append("Agrega un resumen profesional al inicio de tu CV")
        
        if "skills" in missing_elements:
            suggestions.append("Incluye una sección de habilidades")
        
        # Sugerencias basadas en feedback
        for feedback in feedback_by_section.values():
            if feedback.priority == "high" and feedback.weaknesses:
                suggestions.extend(feedback.suggestions[:2])
        
        return list(set(suggestions))[:10]  # Máximo 10 sugerencias únicas




