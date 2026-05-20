"""
Sistema de análisis antes/después
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class BeforeAfterComparison:
    """Comparación antes/después"""
    id: str
    user_id: str
    before_image_url: str
    after_image_url: str
    time_interval_days: int
    metrics_comparison: Dict
    improvement_percentage: float
    key_changes: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "before_image_url": self.before_image_url,
            "after_image_url": self.after_image_url,
            "time_interval_days": self.time_interval_days,
            "metrics_comparison": self.metrics_comparison,
            "improvement_percentage": self.improvement_percentage,
            "key_changes": self.key_changes,
            "created_at": self.created_at
        }


class BeforeAfterAnalysis:
    """Sistema de análisis antes/después"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.comparisons: Dict[str, List[BeforeAfterComparison]] = {}  # user_id -> [comparisons]
    
    def create_comparison(self, user_id: str, before_image_url: str,
                        after_image_url: str, before_analysis: Dict,
                        after_analysis: Dict, time_interval_days: int) -> BeforeAfterComparison:
        """Crea comparación antes/después"""
        # Comparar métricas
        metrics_comparison = self._compare_metrics(before_analysis, after_analysis)
        
        # Calcular porcentaje de mejora
        improvement = self._calculate_improvement(before_analysis, after_analysis)
        
        # Identificar cambios clave
        key_changes = self._identify_key_changes(before_analysis, after_analysis)
        
        comparison = BeforeAfterComparison(
            id=str(uuid.uuid4()),
            user_id=user_id,
            before_image_url=before_image_url,
            after_image_url=after_image_url,
            time_interval_days=time_interval_days,
            metrics_comparison=metrics_comparison,
            improvement_percentage=improvement,
            key_changes=key_changes
        )
        
        if user_id not in self.comparisons:
            self.comparisons[user_id] = []
        
        self.comparisons[user_id].append(comparison)
        return comparison
    
    def _compare_metrics(self, before: Dict, after: Dict) -> Dict:
        """Compara métricas entre antes y después"""
        before_scores = before.get("quality_scores", {})
        after_scores = after.get("quality_scores", {})
        
        comparison = {}
        
        # Comparar cada métrica
        for metric in ["overall_score", "hydration_score", "texture_score", "sebum_level"]:
            before_val = before_scores.get(metric, 0)
            after_val = after_scores.get(metric, 0)
            
            change = after_val - before_val
            percentage_change = (change / before_val * 100) if before_val > 0 else 0
            
            comparison[metric] = {
                "before": before_val,
                "after": after_val,
                "change": change,
                "percentage_change": percentage_change
            }
        
        return comparison
    
    def _calculate_improvement(self, before: Dict, after: Dict) -> float:
        """Calcula porcentaje de mejora general"""
        before_scores = before.get("quality_scores", {})
        after_scores = after.get("quality_scores", {})
        
        before_overall = before_scores.get("overall_score", 0)
        after_overall = after_scores.get("overall_score", 0)
        
        if before_overall > 0:
            improvement = ((after_overall - before_overall) / before_overall) * 100
            return float(improvement)
        
        return 0.0
    
    def _identify_key_changes(self, before: Dict, after: Dict) -> List[str]:
        """Identifica cambios clave"""
        changes = []
        
        metrics_comparison = self._compare_metrics(before, after)
        
        for metric, data in metrics_comparison.items():
            change = data["percentage_change"]
            
            if change > 10:
                metric_name = metric.replace("_", " ").title()
                changes.append(f"{metric_name} mejoró {change:.1f}%")
            elif change < -10:
                metric_name = metric.replace("_", " ").title()
                changes.append(f"{metric_name} empeoró {abs(change):.1f}%")
        
        return changes
    
    def get_user_comparisons(self, user_id: str) -> List[BeforeAfterComparison]:
        """Obtiene comparaciones del usuario"""
        return self.comparisons.get(user_id, [])
    
    def get_comparison(self, user_id: str, comparison_id: str) -> Optional[BeforeAfterComparison]:
        """Obtiene una comparación específica"""
        user_comparisons = self.comparisons.get(user_id, [])
        
        for comparison in user_comparisons:
            if comparison.id == comparison_id:
                return comparison
        
        return None






