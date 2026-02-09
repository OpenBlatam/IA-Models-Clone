"""
Servicio de Análisis Genético y Predisposición - Sistema completo de genética
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class GeneticRiskLevel(str, Enum):
    """Niveles de riesgo genético"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class GeneticPredispositionService:
    """Servicio de análisis genético y predisposición"""
    
    def __init__(self):
        """Inicializa el servicio de genética"""
        pass
    
    def analyze_genetic_data(
        self,
        user_id: str,
        genetic_data: Dict
    ) -> Dict:
        """
        Analiza datos genéticos
        
        Args:
            user_id: ID del usuario
            genetic_data: Datos genéticos
        
        Returns:
            Análisis genético
        """
        analysis = {
            "user_id": user_id,
            "analysis_id": f"genetic_{datetime.now().timestamp()}",
            "genetic_markers": self._analyze_markers(genetic_data),
            "addiction_predisposition": self._calculate_predisposition(genetic_data),
            "risk_factors": self._identify_genetic_risk_factors(genetic_data),
            "protective_factors": self._identify_protective_factors(genetic_data),
            "recommendations": self._generate_genetic_recommendations(genetic_data),
            "analyzed_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def predict_genetic_risk(
        self,
        user_id: str,
        genetic_profile: Dict,
        addiction_type: str
    ) -> Dict:
        """
        Predice riesgo genético
        
        Args:
            user_id: ID del usuario
            genetic_profile: Perfil genético
            addiction_type: Tipo de adicción
        
        Returns:
            Predicción de riesgo genético
        """
        risk_score = self._calculate_genetic_risk_score(genetic_profile, addiction_type)
        
        return {
            "user_id": user_id,
            "addiction_type": addiction_type,
            "genetic_risk_score": risk_score,
            "risk_level": self._determine_risk_level(risk_score),
            "confidence": 0.75,
            "genetic_factors": self._identify_relevant_factors(genetic_profile, addiction_type),
            "recommendations": self._generate_risk_recommendations(risk_score),
            "predicted_at": datetime.now().isoformat()
        }
    
    def get_personalized_treatment_recommendations(
        self,
        user_id: str,
        genetic_analysis: Dict
    ) -> List[Dict]:
        """
        Obtiene recomendaciones de tratamiento personalizadas basadas en genética
        
        Args:
            user_id: ID del usuario
            genetic_analysis: Análisis genético
        
        Returns:
            Recomendaciones de tratamiento
        """
        recommendations = []
        
        risk_level = genetic_analysis.get("addiction_predisposition", {}).get("risk_level", "moderate")
        
        if risk_level == GeneticRiskLevel.HIGH or risk_level == GeneticRiskLevel.VERY_HIGH:
            recommendations.append({
                "type": "intensive_monitoring",
                "priority": "high",
                "description": "Monitoreo intensivo recomendado debido a predisposición genética"
            })
            recommendations.append({
                "type": "early_intervention",
                "priority": "high",
                "description": "Intervención temprana recomendada"
            })
        
        return recommendations
    
    def _analyze_markers(self, genetic_data: Dict) -> List[Dict]:
        """Analiza marcadores genéticos"""
        markers = genetic_data.get("markers", [])
        return [
            {
                "marker": m.get("name"),
                "variant": m.get("variant"),
                "significance": m.get("significance", "unknown")
            }
            for m in markers
        ]
    
    def _calculate_predisposition(self, genetic_data: Dict) -> Dict:
        """Calcula predisposición genética"""
        markers = genetic_data.get("markers", [])
        
        risk_score = 0.5  # Base
        
        # Ajustar basado en marcadores
        for marker in markers:
            significance = marker.get("significance", "unknown")
            if significance == "high_risk":
                risk_score += 0.15
            elif significance == "moderate_risk":
                risk_score += 0.08
        
        risk_score = min(1.0, risk_score)
        
        return {
            "risk_score": round(risk_score, 3),
            "risk_level": self._determine_risk_level(risk_score),
            "confidence": 0.75
        }
    
    def _identify_genetic_risk_factors(self, genetic_data: Dict) -> List[str]:
        """Identifica factores de riesgo genéticos"""
        risk_factors = []
        
        markers = genetic_data.get("markers", [])
        for marker in markers:
            if marker.get("significance") == "high_risk":
                risk_factors.append(marker.get("name", "Unknown marker"))
        
        return risk_factors
    
    def _identify_protective_factors(self, genetic_data: Dict) -> List[str]:
        """Identifica factores protectores genéticos"""
        protective_factors = []
        
        markers = genetic_data.get("markers", [])
        for marker in markers:
            if marker.get("significance") == "protective":
                protective_factors.append(marker.get("name", "Unknown marker"))
        
        return protective_factors
    
    def _generate_genetic_recommendations(self, genetic_data: Dict) -> List[str]:
        """Genera recomendaciones basadas en genética"""
        recommendations = []
        
        predisposition = self._calculate_predisposition(genetic_data)
        risk_level = predisposition.get("risk_level")
        
        if risk_level in [GeneticRiskLevel.HIGH, GeneticRiskLevel.VERY_HIGH]:
            recommendations.append("Predisposición genética alta detectada. Monitoreo cercano recomendado")
        
        return recommendations
    
    def _calculate_genetic_risk_score(self, profile: Dict, addiction_type: str) -> float:
        """Calcula puntuación de riesgo genético"""
        # Lógica simplificada
        base_score = 0.5
        
        # Ajustar por tipo de adicción
        type_multipliers = {
            "alcohol": 1.0,
            "tobacco": 0.9,
            "drugs": 1.1
        }
        
        multiplier = type_multipliers.get(addiction_type, 1.0)
        return min(1.0, base_score * multiplier)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determina nivel de riesgo"""
        if risk_score >= 0.8:
            return GeneticRiskLevel.VERY_HIGH
        elif risk_score >= 0.6:
            return GeneticRiskLevel.HIGH
        elif risk_score >= 0.4:
            return GeneticRiskLevel.MODERATE
        else:
            return GeneticRiskLevel.LOW
    
    def _identify_relevant_factors(self, profile: Dict, addiction_type: str) -> List[str]:
        """Identifica factores relevantes"""
        return []
    
    def _generate_risk_recommendations(self, risk_score: float) -> List[str]:
        """Genera recomendaciones basadas en riesgo"""
        recommendations = []
        
        if risk_score >= 0.7:
            recommendations.append("Riesgo genético elevado. Considera estrategias preventivas adicionales")
        
        return recommendations

