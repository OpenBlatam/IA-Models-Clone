"""
Sistema de análisis comparativo con otros usuarios
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class ComparativeInsight:
    """Insight comparativo"""
    metric: str
    user_value: float
    average_value: float
    percentile: float  # 0-100
    comparison: str  # "above_average", "average", "below_average"
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric": self.metric,
            "user_value": self.user_value,
            "average_value": self.average_value,
            "percentile": self.percentile,
            "comparison": self.comparison,
            "recommendation": self.recommendation
        }


@dataclass
class ComparativeAnalysis:
    """Análisis comparativo"""
    user_id: str
    insights: List[ComparativeInsight]
    overall_percentile: float
    peer_group: str  # "similar_age", "similar_skin_type", "general"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "insights": [i.to_dict() for i in self.insights],
            "overall_percentile": self.overall_percentile,
            "peer_group": self.peer_group,
            "created_at": self.created_at
        }


class ComparativeAnalysisSystem:
    """Sistema de análisis comparativo"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.user_data: Dict[str, Dict] = {}  # user_id -> user_data
    
    def add_user_data(self, user_id: str, skin_analysis: Dict, age: Optional[int] = None,
                     skin_type: Optional[str] = None):
        """Agrega datos de usuario para comparación"""
        self.user_data[user_id] = {
            "skin_analysis": skin_analysis,
            "age": age,
            "skin_type": skin_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def compare_with_peers(self, user_id: str, peer_group: str = "general") -> ComparativeAnalysis:
        """Compara usuario con sus pares"""
        user_data = self.user_data.get(user_id)
        if not user_data:
            raise ValueError("User data not found")
        
        user_scores = user_data["skin_analysis"].get("quality_scores", {})
        
        # Filtrar grupo de pares
        peer_data = self._get_peer_group(peer_group, user_id)
        
        if not peer_data:
            raise ValueError("Insufficient peer data")
        
        insights = []
        
        # Comparar métricas
        for metric in ["overall_score", "hydration_score", "texture_score"]:
            user_value = user_scores.get(metric, 0)
            peer_values = [
                d["skin_analysis"].get("quality_scores", {}).get(metric, 0)
                for d in peer_data
            ]
            
            if peer_values:
                avg_value = statistics.mean(peer_values)
                percentile = self._calculate_percentile(user_value, peer_values)
                
                if percentile >= 75:
                    comparison = "above_average"
                    recommendation = f"Excelente {metric}. Estás en el top 25%."
                elif percentile >= 50:
                    comparison = "average"
                    recommendation = f"{metric} promedio. Hay espacio para mejorar."
                else:
                    comparison = "below_average"
                    recommendation = f"{metric} por debajo del promedio. Considera ajustar tu rutina."
                
                insight = ComparativeInsight(
                    metric=metric,
                    user_value=user_value,
                    average_value=avg_value,
                    percentile=percentile,
                    comparison=comparison,
                    recommendation=recommendation
                )
                insights.append(insight)
        
        # Percentil general
        overall_percentile = statistics.mean([i.percentile for i in insights]) if insights else 50.0
        
        return ComparativeAnalysis(
            user_id=user_id,
            insights=insights,
            overall_percentile=overall_percentile,
            peer_group=peer_group
        )
    
    def _get_peer_group(self, peer_group: str, user_id: str) -> List[Dict]:
        """Obtiene grupo de pares"""
        user_data = self.user_data.get(user_id, {})
        
        if peer_group == "similar_age":
            user_age = user_data.get("age")
            if user_age:
                age_range = (user_age - 5, user_age + 5)
                return [
                    d for uid, d in self.user_data.items()
                    if uid != user_id and age_range[0] <= d.get("age", 0) <= age_range[1]
                ]
        
        elif peer_group == "similar_skin_type":
            user_skin_type = user_data.get("skin_type")
            if user_skin_type:
                return [
                    d for uid, d in self.user_data.items()
                    if uid != user_id and d.get("skin_type") == user_skin_type
                ]
        
        # General - todos los usuarios excepto el actual
        return [
            d for uid, d in self.user_data.items()
            if uid != user_id
        ]
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calcula percentil"""
        if not values:
            return 50.0
        
        sorted_values = sorted(values)
        count_below = sum(1 for v in sorted_values if v < value)
        percentile = (count_below / len(sorted_values)) * 100
        
        return float(percentile)






