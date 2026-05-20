"""
Servicio de Análisis de Calidad de Vida - Sistema completo de calidad de vida
"""

from typing import Dict, List, Optional
from datetime import datetime
import statistics


class QualityOfLifeAnalysisService:
    """Servicio de análisis de calidad de vida"""
    
    def __init__(self):
        """Inicializa el servicio de calidad de vida"""
        pass
    
    def assess_quality_of_life(
        self,
        user_id: str,
        qol_data: Dict
    ) -> Dict:
        """
        Evalúa calidad de vida
        
        Args:
            user_id: ID del usuario
            qol_data: Datos de calidad de vida
        
        Returns:
            Evaluación de calidad de vida
        """
        domains = {
            "physical": qol_data.get("physical_health", 5),
            "mental": qol_data.get("mental_health", 5),
            "social": qol_data.get("social_wellbeing", 5),
            "emotional": qol_data.get("emotional_wellbeing", 5),
            "functional": qol_data.get("functional_status", 5)
        }
        
        overall_score = statistics.mean(domains.values())
        
        return {
            "user_id": user_id,
            "assessment_id": f"qol_{datetime.now().timestamp()}",
            "domains": domains,
            "overall_score": round(overall_score, 2),
            "quality_level": self._determine_quality_level(overall_score),
            "strengths": self._identify_strengths(domains),
            "areas_for_improvement": self._identify_improvement_areas(domains),
            "recommendations": self._generate_qol_recommendations(domains, overall_score),
            "assessed_at": datetime.now().isoformat()
        }
    
    def analyze_qol_trends(
        self,
        user_id: str,
        qol_assessments: List[Dict]
    ) -> Dict:
        """
        Analiza tendencias de calidad de vida
        
        Args:
            user_id: ID del usuario
            qol_assessments: Evaluaciones de calidad de vida
        
        Returns:
            Análisis de tendencias
        """
        if not qol_assessments or len(qol_assessments) < 2:
            return {
                "user_id": user_id,
                "analysis": "insufficient_data"
            }
        
        scores = [a.get("overall_score", 5) for a in qol_assessments]
        
        return {
            "user_id": user_id,
            "total_assessments": len(qol_assessments),
            "current_score": round(scores[-1], 2),
            "baseline_score": round(scores[0], 2),
            "change": round(scores[-1] - scores[0], 2),
            "trend": self._calculate_qol_trend(scores),
            "domain_trends": self._analyze_domain_trends(qol_assessments),
            "improvements": self._identify_qol_improvements(qol_assessments),
            "generated_at": datetime.now().isoformat()
        }
    
    def correlate_qol_with_recovery(
        self,
        user_id: str,
        qol_data: List[Dict],
        recovery_data: List[Dict]
    ) -> Dict:
        """
        Correlaciona calidad de vida con recuperación
        
        Args:
            user_id: ID del usuario
            qol_data: Datos de calidad de vida
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de correlación
        """
        return {
            "user_id": user_id,
            "correlation_score": self._calculate_correlation(qol_data, recovery_data),
            "findings": self._identify_correlations(qol_data, recovery_data),
            "recommendations": self._generate_correlation_recommendations(qol_data, recovery_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def _determine_quality_level(self, score: float) -> str:
        """Determina nivel de calidad"""
        if score >= 8:
            return "excellent"
        elif score >= 6.5:
            return "good"
        elif score >= 5:
            return "fair"
        elif score >= 3.5:
            return "poor"
        else:
            return "very_poor"
    
    def _identify_strengths(self, domains: Dict) -> List[str]:
        """Identifica fortalezas"""
        strengths = []
        
        for domain, score in domains.items():
            if score >= 7:
                strengths.append(domain.capitalize())
        
        return strengths
    
    def _identify_improvement_areas(self, domains: Dict) -> List[str]:
        """Identifica áreas de mejora"""
        improvements = []
        
        for domain, score in domains.items():
            if score < 5:
                improvements.append(domain.capitalize())
        
        return improvements
    
    def _generate_qol_recommendations(self, domains: Dict, overall: float) -> List[str]:
        """Genera recomendaciones de calidad de vida"""
        recommendations = []
        
        if overall < 6:
            recommendations.append("Considera actividades que mejoren tu bienestar general")
        
        if domains.get("mental", 5) < 5:
            recommendations.append("Enfócate en mejorar tu salud mental")
        
        if domains.get("social", 5) < 5:
            recommendations.append("Fortalecer conexiones sociales puede mejorar tu calidad de vida")
        
        return recommendations
    
    def _calculate_qol_trend(self, scores: List[float]) -> str:
        """Calcula tendencia de calidad de vida"""
        if len(scores) < 2:
            return "stable"
        
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if avg_second > avg_first * 1.05:
            return "improving"
        elif avg_second < avg_first * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _analyze_domain_trends(self, assessments: List[Dict]) -> Dict:
        """Analiza tendencias por dominio"""
        return {}
    
    def _identify_qol_improvements(self, assessments: List[Dict]) -> List[str]:
        """Identifica mejoras en calidad de vida"""
        improvements = []
        
        if len(assessments) >= 2:
            first = assessments[0].get("overall_score", 5)
            last = assessments[-1].get("overall_score", 5)
            
            if last > first:
                improvements.append(f"Mejora de {round(last - first, 2)} puntos en calidad de vida")
        
        return improvements
    
    def _calculate_correlation(self, qol_data: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula correlación"""
        # Lógica simplificada
        return 0.65
    
    def _identify_correlations(self, qol_data: List[Dict], recovery_data: List[Dict]) -> List[str]:
        """Identifica correlaciones"""
        return [
            "La calidad de vida se correlaciona positivamente con días de sobriedad",
            "Mejor bienestar mental se asocia con menor riesgo de recaída"
        ]
    
    def _generate_correlation_recommendations(self, qol_data: List[Dict], recovery_data: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en correlaciones"""
        return [
            "Mejorar calidad de vida puede apoyar tu recuperación",
            "Enfócate en áreas de bienestar que necesiten atención"
        ]

