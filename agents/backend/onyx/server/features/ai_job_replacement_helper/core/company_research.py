"""
Company Research Service - Investigación de empresas
======================================================

Sistema de investigación profunda de empresas para preparación de entrevistas.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompanyProfile:
    """Perfil de empresa"""
    company_id: str
    name: str
    industry: str
    size: str
    location: str
    website: str
    description: str
    culture: Dict[str, Any] = field(default_factory=dict)
    benefits: List[str] = field(default_factory=list)
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    recent_news: List[Dict[str, Any]] = field(default_factory=list)
    leadership: List[Dict[str, Any]] = field(default_factory=list)
    tech_stack: List[str] = field(default_factory=list)


@dataclass
class InterviewPrep:
    """Preparación para entrevista"""
    company_id: str
    job_title: str
    key_points: List[str]
    questions_to_ask: List[str]
    red_flags: List[str]
    talking_points: List[str]
    research_summary: str


class CompanyResearchService:
    """Servicio de investigación de empresas"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.companies: Dict[str, CompanyProfile] = {}
        logger.info("CompanyResearchService initialized")
    
    def research_company(self, company_name: str) -> CompanyProfile:
        """Investigar empresa"""
        # En producción, esto consultaría APIs reales (Glassdoor, LinkedIn, etc.)
        # Por ahora, simulamos
        
        company_id = f"company_{company_name.lower().replace(' ', '_')}"
        
        if company_id not in self.companies:
            profile = CompanyProfile(
                company_id=company_id,
                name=company_name,
                industry="Technology",
                size="1000-5000 employees",
                location="San Francisco, CA",
                website=f"https://www.{company_name.lower().replace(' ', '')}.com",
                description=f"{company_name} is a leading technology company...",
                culture={
                    "values": ["Innovation", "Collaboration", "Diversity"],
                    "work_life_balance": 4.2,
                    "career_growth": 4.5,
                },
                benefits=[
                    "Health insurance",
                    "401k matching",
                    "Remote work options",
                    "Stock options",
                ],
                recent_news=[
                    {
                        "title": f"{company_name} announces new product",
                        "date": "2024-01-15",
                        "source": "Tech News",
                    }
                ],
                tech_stack=["Python", "React", "AWS", "Docker"],
            )
            
            self.companies[company_id] = profile
        
        return self.companies[company_id]
    
    def prepare_for_interview(
        self,
        company_id: str,
        job_title: str
    ) -> InterviewPrep:
        """Preparar para entrevista específica"""
        company = self.companies.get(company_id)
        if not company:
            raise ValueError(f"Company {company_id} not found")
        
        prep = InterviewPrep(
            company_id=company_id,
            job_title=job_title,
            key_points=self._generate_key_points(company),
            questions_to_ask=self._generate_questions_to_ask(company, job_title),
            red_flags=self._identify_red_flags(company),
            talking_points=self._generate_talking_points(company, job_title),
            research_summary=self._generate_summary(company, job_title),
        )
        
        return prep
    
    def _generate_key_points(self, company: CompanyProfile) -> List[str]:
        """Generar puntos clave sobre la empresa"""
        return [
            f"{company.name} es líder en {company.industry}",
            f"Tamaño: {company.size}",
            f"Valores principales: {', '.join(company.culture.get('values', []))}",
            f"Stack tecnológico: {', '.join(company.tech_stack[:3])}",
        ]
    
    def _generate_questions_to_ask(
        self,
        company: CompanyProfile,
        job_title: str
    ) -> List[str]:
        """Generar preguntas para hacer en la entrevista"""
        return [
            "¿Cómo es un día típico en este rol?",
            "¿Qué oportunidades de crecimiento profesional hay?",
            "¿Cómo es la cultura de trabajo y colaboración?",
            "¿Qué proyectos o iniciativas están priorizando actualmente?",
            "¿Cómo miden el éxito en este rol?",
        ]
    
    def _identify_red_flags(self, company: CompanyProfile) -> List[str]:
        """Identificar posibles red flags"""
        red_flags = []
        
        if company.culture.get("work_life_balance", 5) < 3.0:
            red_flags.append("Bajo balance trabajo-vida según reviews")
        
        if len(company.benefits) < 3:
            red_flags.append("Beneficios limitados")
        
        return red_flags
    
    def _generate_talking_points(
        self,
        company: CompanyProfile,
        job_title: str
    ) -> List[str]:
        """Generar puntos de conversación"""
        return [
            f"Estoy muy interesado en {company.name} porque [razón específica]",
            f"Mi experiencia en {', '.join(company.tech_stack[:2])} se alinea perfectamente",
            f"Valoro {company.culture.get('values', [])[0] if company.culture.get('values') else 'la innovación'}",
            f"Estoy emocionado por contribuir al equipo de {job_title}",
        ]
    
    def _generate_summary(
        self,
        company: CompanyProfile,
        job_title: str
    ) -> str:
        """Generar resumen de investigación"""
        return f"""
        Resumen de {company.name}:
        
        {company.name} es una empresa de {company.industry} con {company.size}.
        La empresa se enfoca en {', '.join(company.culture.get('values', [])[:2])}.
        
        Para el rol de {job_title}, es importante destacar experiencia en
        {', '.join(company.tech_stack[:3])}.
        
        La empresa ofrece beneficios como {', '.join(company.benefits[:3])}.
        """.strip()
    
    def compare_companies(
        self,
        company_ids: List[str]
    ) -> Dict[str, Any]:
        """Comparar múltiples empresas"""
        companies = [self.companies.get(cid) for cid in company_ids if cid in self.companies]
        
        comparison = {
            "companies": [
                {
                    "name": c.name,
                    "industry": c.industry,
                    "size": c.size,
                    "culture_score": c.culture.get("work_life_balance", 0),
                    "benefits_count": len(c.benefits),
                }
                for c in companies
            ],
            "recommendations": self._generate_comparison_recommendations(companies),
        }
        
        return comparison
    
    def _generate_comparison_recommendations(
        self,
        companies: List[CompanyProfile]
    ) -> List[str]:
        """Generar recomendaciones de comparación"""
        if not companies:
            return []
        
        best_culture = max(companies, key=lambda c: c.culture.get("work_life_balance", 0))
        most_benefits = max(companies, key=lambda c: len(c.benefits))
        
        return [
            f"{best_culture.name} tiene el mejor balance trabajo-vida",
            f"{most_benefits.name} ofrece más beneficios",
        ]




