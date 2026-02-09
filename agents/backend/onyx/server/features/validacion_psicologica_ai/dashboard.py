"""
Dashboard y Visualizaciones para Validación Psicológica AI
==========================================================
Generación de datos para dashboards y visualizaciones
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from uuid import UUID
import structlog
from collections import defaultdict

from .models import PsychologicalValidation, PsychologicalProfile, ValidationReport

logger = structlog.get_logger()


class DashboardDataGenerator:
    """Generador de datos para dashboard"""
    
    def __init__(self):
        """Inicializar generador"""
        logger.info("DashboardDataGenerator initialized")
    
    def generate_overview(
        self,
        validations: List[PsychologicalValidation]
    ) -> Dict[str, Any]:
        """
        Generar datos de overview del dashboard
        
        Args:
            validations: Lista de validaciones
            
        Returns:
            Datos de overview
        """
        total_validations = len(validations)
        completed = len([v for v in validations if v.status.value == "completed"])
        in_progress = len([v for v in validations if v.status.value == "in_progress"])
        failed = len([v for v in validations if v.status.value == "failed"])
        
        # Calcular promedio de confianza
        confidence_scores = [
            v.profile.confidence_score
            for v in validations
            if v.profile and v.status.value == "completed"
        ]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Plataformas más usadas
        platform_counts = defaultdict(int)
        for v in validations:
            for platform in v.connected_platforms:
                platform_counts[platform.value] += 1
        
        return {
            "total_validations": total_validations,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "success_rate": completed / total_validations if total_validations > 0 else 0.0,
            "average_confidence": avg_confidence,
            "platform_usage": dict(platform_counts),
            "last_validation": (
                max(v.created_at for v in validations).isoformat()
                if validations else None
            )
        }
    
    def generate_timeline_data(
        self,
        validations: List[PsychologicalValidation],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generar datos de timeline
        
        Args:
            validations: Lista de validaciones
            days: Días a mostrar
            
        Returns:
            Datos de timeline
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_validations = [
            v for v in validations
            if v.created_at >= cutoff_date
        ]
        
        # Agrupar por día
        daily_data = defaultdict(lambda: {
            "count": 0,
            "completed": 0,
            "failed": 0,
            "avg_confidence": []
        })
        
        for v in recent_validations:
            date_key = v.created_at.date().isoformat()
            daily_data[date_key]["count"] += 1
            
            if v.status.value == "completed":
                daily_data[date_key]["completed"] += 1
                if v.profile:
                    daily_data[date_key]["avg_confidence"].append(v.profile.confidence_score)
            elif v.status.value == "failed":
                daily_data[date_key]["failed"] += 1
        
        # Calcular promedios
        timeline = []
        for date_str, data in sorted(daily_data.items()):
            avg_conf = (
                sum(data["avg_confidence"]) / len(data["avg_confidence"])
                if data["avg_confidence"] else 0.0
            )
            timeline.append({
                "date": date_str,
                "total": data["count"],
                "completed": data["completed"],
                "failed": data["failed"],
                "average_confidence": avg_conf
            })
        
        return {
            "timeline": timeline,
            "period_days": days,
            "total_points": len(timeline)
        }
    
    def generate_personality_distribution(
        self,
        profiles: List[PsychologicalProfile]
    ) -> Dict[str, Any]:
        """
        Generar distribución de rasgos de personalidad
        
        Args:
            profiles: Lista de perfiles
            
        Returns:
            Distribución de rasgos
        """
        if not profiles:
            return {"traits": {}, "total_profiles": 0}
        
        trait_averages = defaultdict(list)
        
        for profile in profiles:
            for trait, value in profile.personality_traits.items():
                trait_averages[trait].append(value)
        
        distribution = {}
        for trait, values in trait_averages.items():
            distribution[trait] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std_dev": (
                    (sum((v - sum(values) / len(values))**2 for v in values) / len(values))**0.5
                )
            }
        
        return {
            "traits": distribution,
            "total_profiles": len(profiles)
        }
    
    def generate_sentiment_trends(
        self,
        reports: List[ValidationReport]
    ) -> Dict[str, Any]:
        """
        Generar tendencias de sentimientos
        
        Args:
            reports: Lista de reportes
            
        Returns:
            Tendencias de sentimientos
        """
        if not reports:
            return {"trends": [], "total_reports": 0}
        
        sentiment_data = []
        for report in reports:
            sentiment = report.sentiment_analysis
            if sentiment:
                sentiment_data.append({
                    "date": report.generated_at.date().isoformat(),
                    "overall": sentiment.get("overall_sentiment", "neutral"),
                    "distribution": sentiment.get("sentiment_distribution", {}),
                    "score": sentiment.get("average_score", 0.0)
                })
        
        # Agrupar por sentimiento
        sentiment_counts = defaultdict(int)
        for data in sentiment_data:
            sentiment_counts[data["overall"]] += 1
        
        return {
            "trends": sentiment_data,
            "distribution": dict(sentiment_counts),
            "total_reports": len(reports),
            "positive_ratio": sentiment_counts.get("positive", 0) / len(sentiment_data) if sentiment_data else 0.0,
            "negative_ratio": sentiment_counts.get("negative", 0) / len(sentiment_data) if sentiment_data else 0.0
        }
    
    def generate_platform_insights(
        self,
        validations: List[PsychologicalValidation]
    ) -> Dict[str, Any]:
        """
        Generar insights por plataforma
        
        Args:
            validations: Lista de validaciones
            
        Returns:
            Insights por plataforma
        """
        platform_data = defaultdict(lambda: {
            "count": 0,
            "avg_confidence": [],
            "total_posts": 0,
            "total_engagement": 0
        })
        
        for validation in validations:
            if not validation.report:
                continue
            
            for platform, insights in validation.report.social_media_insights.items():
                platform_data[platform]["count"] += 1
                platform_data[platform]["total_posts"] += insights.get("post_count", 0)
                platform_data[platform]["total_engagement"] += insights.get("total_engagement", 0)
                
                if validation.profile:
                    platform_data[platform]["avg_confidence"].append(
                        validation.profile.confidence_score
                    )
        
        insights = {}
        for platform, data in platform_data.items():
            insights[platform] = {
                "usage_count": data["count"],
                "average_confidence": (
                    sum(data["avg_confidence"]) / len(data["avg_confidence"])
                    if data["avg_confidence"] else 0.0
                ),
                "total_posts_analyzed": data["total_posts"],
                "total_engagement": data["total_engagement"],
                "avg_posts_per_validation": (
                    data["total_posts"] / data["count"]
                    if data["count"] > 0 else 0
                )
            }
        
        return {
            "platforms": insights,
            "total_platforms": len(insights)
        }
    
    def generate_risk_analysis(
        self,
        profiles: List[PsychologicalProfile]
    ) -> Dict[str, Any]:
        """
        Generar análisis de riesgos
        
        Args:
            profiles: Lista de perfiles
            
        Returns:
            Análisis de riesgos
        """
        if not profiles:
            return {"risks": [], "total_profiles": 0}
        
        risk_factors = defaultdict(int)
        high_risk_count = 0
        
        for profile in profiles:
            # Contar factores de riesgo
            for risk in profile.risk_factors:
                risk_factors[risk] += 1
            
            # Detectar perfiles de alto riesgo
            neuroticism = profile.personality_traits.get("neuroticism", 0.5)
            stress_level = profile.emotional_state.get("stress_level", 0.0)
            sentiment = profile.emotional_state.get("overall_sentiment", "neutral")
            
            if neuroticism > 0.7 or stress_level > 0.7 or sentiment == "negative":
                high_risk_count += 1
        
        return {
            "risk_factors": dict(risk_factors),
            "high_risk_profiles": high_risk_count,
            "high_risk_percentage": high_risk_count / len(profiles) if profiles else 0.0,
            "total_profiles": len(profiles)
        }
    
    def generate_complete_dashboard(
        self,
        validations: List[PsychologicalValidation]
    ) -> Dict[str, Any]:
        """
        Generar dashboard completo
        
        Args:
            validations: Lista de validaciones
            
        Returns:
            Datos completos del dashboard
        """
        profiles = [v.profile for v in validations if v.profile]
        reports = [v.report for v in validations if v.report]
        
        return {
            "overview": self.generate_overview(validations),
            "timeline": self.generate_timeline_data(validations),
            "personality_distribution": self.generate_personality_distribution(profiles),
            "sentiment_trends": self.generate_sentiment_trends(reports),
            "platform_insights": self.generate_platform_insights(validations),
            "risk_analysis": self.generate_risk_analysis(profiles),
            "generated_at": datetime.utcnow().isoformat()
        }


# Instancia global del generador
dashboard_generator = DashboardDataGenerator()




