"""
Servicio de Análisis de Motivación Avanzado - Sistema completo de análisis de motivación
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedMotivationAnalysisService:
    """Servicio de análisis de motivación avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de motivación"""
        pass
    
    def assess_motivation(
        self,
        user_id: str,
        motivation_data: Dict
    ) -> Dict:
        """
        Evalúa motivación
        
        Args:
            user_id: ID del usuario
            motivation_data: Datos de motivación
        
        Returns:
            Evaluación de motivación
        """
        return {
            "user_id": user_id,
            "assessment_id": f"motivation_{datetime.now().timestamp()}",
            "motivation_score": self._calculate_motivation_score(motivation_data),
            "motivation_type": self._determine_motivation_type(motivation_data),
            "intrinsic_motivation": motivation_data.get("intrinsic_motivation", 5),
            "extrinsic_motivation": motivation_data.get("extrinsic_motivation", 5),
            "motivation_factors": self._analyze_motivation_factors(motivation_data),
            "recommendations": self._generate_motivation_recommendations(motivation_data),
            "assessed_at": datetime.now().isoformat()
        }
    
    def track_motivation_trends(
        self,
        user_id: str,
        motivation_assessments: List[Dict]
    ) -> Dict:
        """
        Rastrea tendencias de motivación
        
        Args:
            user_id: ID del usuario
            motivation_assessments: Evaluaciones de motivación
        
        Returns:
            Análisis de tendencias
        """
        if not motivation_assessments or len(motivation_assessments) < 2:
            return {
                "user_id": user_id,
                "analysis": "insufficient_data"
            }
        
        scores = [a.get("motivation_score", 0) for a in motivation_assessments]
        
        return {
            "user_id": user_id,
            "total_assessments": len(motivation_assessments),
            "average_score": round(statistics.mean(scores), 2) if scores else 0,
            "trend": self._calculate_motivation_trend(scores),
            "volatility": self._calculate_volatility(scores),
            "motivation_stability": self._assess_stability(scores),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_motivation_drop(
        self,
        user_id: str,
        current_motivation: Dict,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Predice caída de motivación
        
        Args:
            user_id: ID del usuario
            current_motivation: Motivación actual
            historical_data: Datos históricos
        
        Returns:
            Predicción de caída
        """
        drop_probability = self._calculate_drop_probability(current_motivation, historical_data)
        
        return {
            "user_id": user_id,
            "drop_probability": round(drop_probability, 3),
            "risk_level": "high" if drop_probability >= 0.7 else "medium" if drop_probability >= 0.4 else "low",
            "warning_signs": self._identify_warning_signs(current_motivation),
            "prevention_strategies": self._generate_prevention_strategies(drop_probability),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _calculate_motivation_score(self, data: Dict) -> float:
        """Calcula puntuación de motivación"""
        intrinsic = data.get("intrinsic_motivation", 5)
        extrinsic = data.get("extrinsic_motivation", 5)
        
        # Ponderar más la motivación intrínseca
        score = (intrinsic * 0.6) + (extrinsic * 0.4)
        
        return round(max(1, min(10, score)), 2)
    
    def _determine_motivation_type(self, data: Dict) -> str:
        """Determina tipo de motivación"""
        intrinsic = data.get("intrinsic_motivation", 5)
        extrinsic = data.get("extrinsic_motivation", 5)
        
        if intrinsic > extrinsic * 1.2:
            return "intrinsic"
        elif extrinsic > intrinsic * 1.2:
            return "extrinsic"
        else:
            return "mixed"
    
    def _analyze_motivation_factors(self, data: Dict) -> Dict:
        """Analiza factores de motivación"""
        return {
            "goals_clarity": data.get("goals_clarity", 5),
            "progress_perception": data.get("progress_perception", 5),
            "support_level": data.get("support_level", 5),
            "self_efficacy": data.get("self_efficacy", 5)
        }
    
    def _generate_motivation_recommendations(self, data: Dict) -> List[str]:
        """Genera recomendaciones de motivación"""
        recommendations = []
        
        score = self._calculate_motivation_score(data)
        
        if score < 6:
            recommendations.append("Establece objetivos claros y alcanzables")
            recommendations.append("Celebra pequeños logros para mantener la motivación")
        
        return recommendations
    
    def _calculate_motivation_trend(self, scores: List[float]) -> str:
        """Calcula tendencia de motivación"""
        if len(scores) < 2:
            return "stable"
        
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if avg_second > avg_first * 1.1:
            return "increasing"
        elif avg_second < avg_first * 0.9:
            return "decreasing"
        return "stable"
    
    def _calculate_volatility(self, scores: List[float]) -> float:
        """Calcula volatilidad"""
        if len(scores) < 2:
            return 0.0
        
        return round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0
    
    def _assess_stability(self, scores: List[float]) -> str:
        """Evalúa estabilidad"""
        volatility = self._calculate_volatility(scores)
        
        if volatility < 1.0:
            return "stable"
        elif volatility < 2.0:
            return "moderate"
        else:
            return "volatile"
    
    def _calculate_drop_probability(self, current: Dict, history: List[Dict]) -> float:
        """Calcula probabilidad de caída"""
        base_probability = 0.3
        
        current_score = self._calculate_motivation_score(current)
        if current_score < 5:
            base_probability += 0.3
        
        if history:
            historical_scores = [self._calculate_motivation_score(h) for h in history]
            if len(historical_scores) >= 2:
                recent_trend = historical_scores[-1] - historical_scores[-2]
                if recent_trend < -1:
                    base_probability += 0.2
        
        return min(1.0, base_probability)
    
    def _identify_warning_signs(self, current: Dict) -> List[str]:
        """Identifica señales de advertencia"""
        signs = []
        
        score = self._calculate_motivation_score(current)
        if score < 5:
            signs.append("Baja motivación actual")
        
        if current.get("goals_clarity", 5) < 4:
            signs.append("Falta de claridad en objetivos")
        
        return signs
    
    def _generate_prevention_strategies(self, drop_probability: float) -> List[str]:
        """Genera estrategias de prevención"""
        strategies = []
        
        if drop_probability >= 0.7:
            strategies.append("⚠️ Alto riesgo de caída de motivación. Contacta tu sistema de apoyo")
            strategies.append("Revisa y ajusta tus objetivos")
        elif drop_probability >= 0.4:
            strategies.append("Monitorea tu motivación regularmente")
            strategies.append("Celebra tus logros")
        
        return strategies

