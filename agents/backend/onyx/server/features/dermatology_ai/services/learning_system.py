"""
Sistema de aprendizaje automático
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import numpy as np


@dataclass
class LearningInsight:
    """Insight de aprendizaje"""
    type: str  # "pattern", "trend", "anomaly", etc.
    description: str
    confidence: float
    data_points: int
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "type": self.type,
            "description": self.description,
            "confidence": self.confidence,
            "data_points": self.data_points,
            "created_at": self.created_at
        }


class LearningSystem:
    """Sistema de aprendizaje automático"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.user_patterns: Dict[str, Dict] = {}  # user_id -> patterns
        self.global_patterns: Dict[str, Any] = {}
        self.anomalies: List[Dict] = []
    
    def learn_from_data(self, user_id: str, data_points: List[Dict]) -> List[LearningInsight]:
        """
        Aprende de datos del usuario
        
        Args:
            user_id: ID del usuario
            data_points: Puntos de datos
            
        Returns:
            Lista de insights
        """
        insights = []
        
        # Detectar patrones
        patterns = self._detect_patterns(data_points)
        if patterns:
            insights.append(LearningInsight(
                type="pattern",
                description=f"Patrones detectados: {len(patterns)}",
                confidence=0.8,
                data_points=len(data_points)
            ))
            self.user_patterns[user_id] = patterns
        
        # Detectar tendencias
        trends = self._detect_trends(data_points)
        if trends:
            insights.append(LearningInsight(
                type="trend",
                description=f"Tendencias detectadas: {trends.get('direction', 'unknown')}",
                confidence=0.75,
                data_points=len(data_points)
            ))
        
        # Detectar anomalías
        anomalies = self._detect_anomalies(data_points)
        if anomalies:
            insights.append(LearningInsight(
                type="anomaly",
                description=f"Anomalías detectadas: {len(anomalies)}",
                confidence=0.7,
                data_points=len(data_points)
            ))
            self.anomalies.extend(anomalies)
        
        return insights
    
    def _detect_patterns(self, data_points: List[Dict]) -> Dict:
        """Detecta patrones en los datos"""
        if len(data_points) < 3:
            return {}
        
        # Analizar scores
        scores = [dp.get("quality_scores", {}).get("overall_score", 0) for dp in data_points]
        
        patterns = {}
        
        # Patrón: Mejora constante
        if len(scores) >= 3:
            recent_trend = scores[-1] - scores[0]
            if recent_trend > 5:
                patterns["improving_trend"] = True
        
        # Patrón: Estabilidad
        score_variance = np.var(scores)
        if score_variance < 10:
            patterns["stability"] = True
        
        return patterns
    
    def _detect_trends(self, data_points: List[Dict]) -> Dict:
        """Detecta tendencias"""
        if len(data_points) < 2:
            return {}
        
        scores = [dp.get("quality_scores", {}).get("overall_score", 0) for dp in data_points]
        
        if len(scores) < 2:
            return {}
        
        # Regresión lineal simple
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
        
        return {
            "direction": direction,
            "slope": float(slope),
            "strength": abs(slope)
        }
    
    def _detect_anomalies(self, data_points: List[Dict]) -> List[Dict]:
        """Detecta anomalías"""
        if len(data_points) < 3:
            return []
        
        scores = [dp.get("quality_scores", {}).get("overall_score", 0) for dp in data_points]
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        anomalies = []
        for i, (score, data_point) in enumerate(zip(scores, data_points)):
            # Detectar valores atípicos (más de 2 desviaciones estándar)
            if abs(score - mean_score) > 2 * std_score:
                anomalies.append({
                    "index": i,
                    "score": score,
                    "deviation": abs(score - mean_score) / std_score,
                    "data_point": data_point
                })
        
        return anomalies
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Obtiene insights del usuario"""
        patterns = self.user_patterns.get(user_id, {})
        
        return {
            "user_id": user_id,
            "patterns": patterns,
            "anomalies_count": len([a for a in self.anomalies if a.get("user_id") == user_id]),
            "learning_active": len(patterns) > 0
        }
    
    def predict_future_state(self, user_id: str, days_ahead: int = 30) -> Dict:
        """Predice estado futuro"""
        patterns = self.user_patterns.get(user_id, {})
        
        # Placeholder - implementar con modelo predictivo real
        prediction = {
            "user_id": user_id,
            "days_ahead": days_ahead,
            "predicted_score": 75.0,
            "confidence": 0.6,
            "factors": ["historical_trend", "current_state"]
        }
        
        return prediction






