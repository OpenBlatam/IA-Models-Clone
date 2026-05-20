"""
Market Trends Service - Análisis de tendencias del mercado
===========================================================

Sistema para analizar tendencias del mercado laboral.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MarketTrend:
    """Tendencia de mercado"""
    skill: str
    industry: str
    demand_score: float  # 0.0 - 1.0
    growth_rate: float  # Porcentaje de crecimiento
    salary_trend: str  # "increasing", "stable", "decreasing"
    future_outlook: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IndustryAnalysis:
    """Análisis de industria"""
    industry: str
    growth_rate: float
    top_skills: List[Dict[str, Any]]
    job_openings_trend: str
    salary_range: Dict[str, float]
    competition_level: str  # "low", "medium", "high"
    recommendations: List[str]


class MarketTrendsService:
    """Servicio de análisis de tendencias"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.trends_cache: Dict[str, List[MarketTrend]] = {}
        logger.info("MarketTrendsService initialized")
    
    def analyze_skill_trend(
        self,
        skill: str,
        industry: Optional[str] = None
    ) -> MarketTrend:
        """Analizar tendencia de una habilidad"""
        # En producción, esto consultaría APIs reales (LinkedIn, Indeed, etc.)
        # Por ahora, simulamos
        
        trend = MarketTrend(
            skill=skill,
            industry=industry or "Technology",
            demand_score=0.75,  # Simulado
            growth_rate=15.5,  # Porcentaje
            salary_trend="increasing",
            future_outlook="Strong demand expected to continue",
        )
        
        return trend
    
    def analyze_industry(
        self,
        industry: str
    ) -> IndustryAnalysis:
        """Analizar industria completa"""
        # En producción, esto analizaría datos reales
        analysis = IndustryAnalysis(
            industry=industry,
            growth_rate=12.3,
            top_skills=[
                {"skill": "Python", "demand": 0.9, "growth": 20.0},
                {"skill": "React", "demand": 0.85, "growth": 18.0},
                {"skill": "AWS", "demand": 0.8, "growth": 25.0},
            ],
            job_openings_trend="increasing",
            salary_range={
                "min": 80000,
                "median": 120000,
                "max": 180000,
            },
            competition_level="medium",
            recommendations=[
                f"La industria {industry} está en crecimiento",
                "Enfócate en habilidades técnicas en alta demanda",
                "Considera certificaciones para destacar",
            ],
        )
        
        return analysis
    
    def get_emerging_skills(
        self,
        industry: str,
        timeframe_days: int = 90
    ) -> List[Dict[str, Any]]:
        """Obtener habilidades emergentes"""
        # En producción, esto analizaría datos de los últimos N días
        emerging = [
            {
                "skill": "AI/ML",
                "growth_rate": 45.0,
                "demand_increase": 0.6,
                "reason": "Rapid adoption of AI technologies",
            },
            {
                "skill": "Cloud Architecture",
                "growth_rate": 35.0,
                "demand_increase": 0.5,
                "reason": "Cloud migration continues to accelerate",
            },
            {
                "skill": "DevOps",
                "growth_rate": 30.0,
                "demand_increase": 0.4,
                "reason": "Focus on automation and CI/CD",
            },
        ]
        
        return emerging
    
    def predict_skill_demand(
        self,
        skill: str,
        months_ahead: int = 12
    ) -> Dict[str, Any]:
        """Predecir demanda futura de habilidad"""
        # En producción, esto usaría modelos de ML
        current_demand = 0.75
        growth_rate = 0.15  # 15% anual
        
        predicted_demand = current_demand * (1 + growth_rate * (months_ahead / 12))
        predicted_demand = min(1.0, predicted_demand)  # Cap at 1.0
        
        return {
            "skill": skill,
            "current_demand": current_demand,
            "predicted_demand": round(predicted_demand, 2),
            "months_ahead": months_ahead,
            "confidence": 0.7,
            "factors": [
                "Industry adoption rate",
                "Technology maturity",
                "Market competition",
            ],
        }
    
    def compare_skills(
        self,
        skills: List[str],
        industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comparar múltiples habilidades"""
        trends = [self.analyze_skill_trend(skill, industry) for skill in skills]
        
        # Encontrar mejor opción
        best_skill = max(trends, key=lambda t: t.demand_score * t.growth_rate)
        
        return {
            "skills": [
                {
                    "skill": t.skill,
                    "demand_score": t.demand_score,
                    "growth_rate": t.growth_rate,
                    "salary_trend": t.salary_trend,
                }
                for t in trends
            ],
            "recommendation": {
                "best_skill": best_skill.skill,
                "reason": f"Highest combined demand and growth: {best_skill.demand_score * 100:.0f}% demand, {best_skill.growth_rate}% growth",
            },
        }
    
    def get_market_insights(
        self,
        user_skills: List[str],
        target_industry: str
    ) -> Dict[str, Any]:
        """Obtener insights de mercado personalizados"""
        # Analizar habilidades del usuario
        skill_trends = [self.analyze_skill_trend(skill, target_industry) for skill in user_skills]
        
        # Analizar industria
        industry_analysis = self.analyze_industry(target_industry)
        
        # Calcular fit del usuario
        avg_demand = sum(t.demand_score for t in skill_trends) / len(skill_trends) if skill_trends else 0.0
        
        insights = {
            "user_fit": {
                "score": avg_demand,
                "interpretation": "strong" if avg_demand > 0.7 else "moderate" if avg_demand > 0.5 else "weak",
            },
            "skill_analysis": [
                {
                    "skill": t.skill,
                    "demand": t.demand_score,
                    "trend": t.salary_trend,
                }
                for t in skill_trends
            ],
            "industry_overview": {
                "growth_rate": industry_analysis.growth_rate,
                "competition": industry_analysis.competition_level,
                "salary_range": industry_analysis.salary_range,
            },
            "recommendations": industry_analysis.recommendations + [
                f"Tus habilidades tienen una demanda {'alta' if avg_demand > 0.7 else 'moderada' if avg_demand > 0.5 else 'baja'} en {target_industry}",
            ],
        }
        
        return insights




