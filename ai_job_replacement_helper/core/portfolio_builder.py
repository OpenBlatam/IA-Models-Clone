"""
Portfolio Builder Service - Constructor de portafolio profesional
==================================================================

Sistema para construir y gestionar portafolio profesional.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProjectType(str, Enum):
    """Tipos de proyecto"""
    WEB_APP = "web_app"
    MOBILE_APP = "mobile_app"
    DATA_SCIENCE = "data_science"
    OPEN_SOURCE = "open_source"
    FREELANCE = "freelance"
    PERSONAL = "personal"


@dataclass
class Project:
    """Proyecto del portafolio"""
    id: str
    title: str
    description: str
    project_type: ProjectType
    technologies: List[str]
    github_url: Optional[str] = None
    live_url: Optional[str] = None
    images: List[str] = field(default_factory=list)
    featured: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Portfolio:
    """Portafolio completo"""
    user_id: str
    projects: List[Project]
    bio: str
    skills: List[str]
    contact_info: Dict[str, str]
    custom_domain: Optional[str] = None
    theme: str = "default"
    views: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class PortfolioBuilderService:
    """Servicio de construcción de portafolio"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.portfolios: Dict[str, Portfolio] = {}
        logger.info("PortfolioBuilderService initialized")
    
    def create_portfolio(
        self,
        user_id: str,
        bio: str,
        skills: List[str],
        contact_info: Dict[str, str]
    ) -> Portfolio:
        """Crear portafolio"""
        portfolio = Portfolio(
            user_id=user_id,
            projects=[],
            bio=bio,
            skills=skills,
            contact_info=contact_info,
        )
        
        self.portfolios[user_id] = portfolio
        
        logger.info(f"Portfolio created for user {user_id}")
        return portfolio
    
    def add_project(
        self,
        user_id: str,
        title: str,
        description: str,
        project_type: ProjectType,
        technologies: List[str],
        github_url: Optional[str] = None,
        live_url: Optional[str] = None,
        featured: bool = False
    ) -> Project:
        """Agregar proyecto al portafolio"""
        portfolio = self.portfolios.get(user_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found for user {user_id}")
        
        project_id = f"project_{user_id}_{int(datetime.now().timestamp())}"
        
        project = Project(
            id=project_id,
            title=title,
            description=description,
            project_type=project_type,
            technologies=technologies,
            github_url=github_url,
            live_url=live_url,
            featured=featured,
        )
        
        portfolio.projects.append(project)
        portfolio.last_updated = datetime.now()
        
        logger.info(f"Project added to portfolio: {project_id}")
        return project
    
    def generate_portfolio_html(self, user_id: str) -> str:
        """Generar HTML del portafolio"""
        portfolio = self.portfolios.get(user_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found for user {user_id}")
        
        # En producción, esto generaría HTML real
        # Por ahora, retornamos un template básico
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{portfolio.user_id}'s Portfolio</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .bio {{ margin-bottom: 30px; }}
        .project {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; }}
        .skills {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .skill {{ background: #007bff; color: white; padding: 5px 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{portfolio.user_id}'s Portfolio</h1>
    <div class="bio">
        <p>{portfolio.bio}</p>
    </div>
    <div class="skills">
        {''.join(f'<span class="skill">{skill}</span>' for skill in portfolio.skills)}
    </div>
    <h2>Projects</h2>
    {''.join(self._generate_project_html(p) for p in portfolio.projects)}
</body>
</html>
        """.strip()
        
        return html
    
    def _generate_project_html(self, project: Project) -> str:
        """Generar HTML de un proyecto"""
        return f"""
    <div class="project">
        <h3>{project.title}</h3>
        <p>{project.description}</p>
        <p><strong>Technologies:</strong> {', '.join(project.technologies)}</p>
        {f'<p><a href="{project.github_url}">GitHub</a></p>' if project.github_url else ''}
        {f'<p><a href="{project.live_url}">Live Demo</a></p>' if project.live_url else ''}
    </div>
        """.strip()
    
    def analyze_portfolio(self, user_id: str) -> Dict[str, Any]:
        """Analizar portafolio y dar recomendaciones"""
        portfolio = self.portfolios.get(user_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found for user {user_id}")
        
        analysis = {
            "total_projects": len(portfolio.projects),
            "featured_projects": sum(1 for p in portfolio.projects if p.featured),
            "technologies_used": set(),
            "project_types": {},
            "recommendations": [],
        }
        
        # Analizar tecnologías
        for project in portfolio.projects:
            analysis["technologies_used"].update(project.technologies)
            project_type = project.project_type.value
            analysis["project_types"][project_type] = analysis["project_types"].get(project_type, 0) + 1
        
        analysis["technologies_used"] = list(analysis["technologies_used"])
        
        # Generar recomendaciones
        if len(portfolio.projects) < 3:
            analysis["recommendations"].append(
                "Agrega más proyectos para mostrar tu experiencia"
            )
        
        if not any(p.featured for p in portfolio.projects):
            analysis["recommendations"].append(
                "Marca al menos un proyecto como destacado"
            )
        
        if not any(p.live_url for p in portfolio.projects):
            analysis["recommendations"].append(
                "Agrega enlaces a demos en vivo de tus proyectos"
            )
        
        return analysis
    
    def export_portfolio(self, user_id: str, format: str = "json") -> Dict[str, Any]:
        """Exportar portafolio"""
        portfolio = self.portfolios.get(user_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found for user {user_id}")
        
        if format == "json":
            return {
                "user_id": portfolio.user_id,
                "bio": portfolio.bio,
                "skills": portfolio.skills,
                "contact_info": portfolio.contact_info,
                "projects": [
                    {
                        "title": p.title,
                        "description": p.description,
                        "type": p.project_type.value,
                        "technologies": p.technologies,
                        "github_url": p.github_url,
                        "live_url": p.live_url,
                        "featured": p.featured,
                    }
                    for p in portfolio.projects
                ],
            }
        
        raise ValueError(f"Unsupported format: {format}")




