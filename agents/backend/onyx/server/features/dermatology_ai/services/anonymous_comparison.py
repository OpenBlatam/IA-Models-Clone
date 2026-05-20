"""
Sistema de análisis comparativo anónimo con otros usuarios
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import statistics


@dataclass
class ComparisonGroup:
    """Grupo de comparación"""
    group_id: str
    criteria: Dict  # {"age_range": "30-40", "skin_type": "oily", "concerns": ["acne"]}
    user_count: int
    average_metrics: Dict
    percentile_data: Dict
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "group_id": self.group_id,
            "criteria": self.criteria,
            "user_count": self.user_count,
            "average_metrics": self.average_metrics,
            "percentile_data": self.percentile_data
        }


@dataclass
class AnonymousComparisonReport:
    """Reporte de comparación anónima"""
    id: str
    user_id: str
    user_metrics: Dict
    comparison_groups: List[ComparisonGroup]
    percentile_rankings: Dict[str, float]
    insights: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_metrics": self.user_metrics,
            "comparison_groups": [g.to_dict() for g in self.comparison_groups],
            "percentile_rankings": self.percentile_rankings,
            "insights": self.insights,
            "created_at": self.created_at
        }


class AnonymousComparison:
    """Sistema de comparación anónima"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[AnonymousComparisonReport]] = {}
        self.anonymous_data: Dict[str, List[Dict]] = {}  # group_id -> [metrics]
    
    def add_anonymous_data(self, group_id: str, metrics: Dict):
        """Agrega datos anónimos para comparación"""
        if group_id not in self.anonymous_data:
            self.anonymous_data[group_id] = []
        
        self.anonymous_data[group_id].append(metrics)
    
    def generate_comparison(self, user_id: str, user_metrics: Dict,
                           user_profile: Dict) -> AnonymousComparisonReport:
        """Genera comparación anónima"""
        
        # Crear grupos de comparación basados en perfil
        comparison_groups = []
        percentile_rankings = {}
        insights = []
        
        # Grupo 1: Mismo tipo de piel
        skin_type = user_profile.get("skin_type", "unknown")
        skin_group_id = f"skin_type_{skin_type}"
        
        if skin_group_id in self.anonymous_data:
            group_data = self.anonymous_data[skin_group_id]
            avg_metrics = self._calculate_average_metrics(group_data)
            percentile = self._calculate_percentile(user_metrics, group_data)
            
            comparison_groups.append(ComparisonGroup(
                group_id=skin_group_id,
                criteria={"skin_type": skin_type},
                user_count=len(group_data),
                average_metrics=avg_metrics,
                percentile_data={"user_percentile": percentile}
            ))
            
            percentile_rankings["skin_type_group"] = percentile
            
            if percentile > 75:
                insights.append(f"Estás en el top 25% para tu tipo de piel ({skin_type})")
            elif percentile < 25:
                insights.append(f"Hay espacio para mejora comparado con otros con piel {skin_type}")
        
        # Grupo 2: Mismo rango de edad
        age_range = user_profile.get("age_range", "unknown")
        age_group_id = f"age_{age_range}"
        
        if age_group_id in self.anonymous_data:
            group_data = self.anonymous_data[age_group_id]
            avg_metrics = self._calculate_average_metrics(group_data)
            percentile = self._calculate_percentile(user_metrics, group_data)
            
            comparison_groups.append(ComparisonGroup(
                group_id=age_group_id,
                criteria={"age_range": age_range},
                user_count=len(group_data),
                average_metrics=avg_metrics,
                percentile_data={"user_percentile": percentile}
            ))
            
            percentile_rankings["age_group"] = percentile
        
        # Grupo 3: Mismas preocupaciones
        concerns = user_profile.get("concerns", [])
        if concerns:
            concerns_key = "_".join(sorted(concerns))
            concerns_group_id = f"concerns_{concerns_key}"
            
            if concerns_group_id in self.anonymous_data:
                group_data = self.anonymous_data[concerns_group_id]
                avg_metrics = self._calculate_average_metrics(group_data)
                percentile = self._calculate_percentile(user_metrics, group_data)
                
                comparison_groups.append(ComparisonGroup(
                    group_id=concerns_group_id,
                    criteria={"concerns": concerns},
                    user_count=len(group_data),
                    average_metrics=avg_metrics,
                    percentile_data={"user_percentile": percentile}
                ))
                
                percentile_rankings["concerns_group"] = percentile
        
        # Insights generales
        if not insights:
            avg_percentile = statistics.mean(percentile_rankings.values()) if percentile_rankings else 50.0
            if avg_percentile > 70:
                insights.append("Tu piel está por encima del promedio en comparación con usuarios similares")
            elif avg_percentile < 30:
                insights.append("Hay oportunidades de mejora comparado con usuarios similares")
            else:
                insights.append("Tu piel está cerca del promedio para usuarios similares")
        
        report = AnonymousComparisonReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            user_metrics=user_metrics,
            comparison_groups=comparison_groups,
            percentile_rankings=percentile_rankings,
            insights=insights
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _calculate_average_metrics(self, data_points: List[Dict]) -> Dict:
        """Calcula métricas promedio"""
        if not data_points:
            return {}
        
        metrics = {}
        for point in data_points:
            for key, value in point.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        averages = {}
        for key, values in metrics.items():
            averages[key] = statistics.mean(values)
        
        return averages
    
    def _calculate_percentile(self, user_metrics: Dict, group_data: List[Dict]) -> float:
        """Calcula percentil del usuario"""
        if not group_data:
            return 50.0
        
        # Calcular percentil para cada métrica y promediar
        percentiles = []
        
        for metric_name, user_value in user_metrics.items():
            if not isinstance(user_value, (int, float)):
                continue
            
            group_values = [d.get(metric_name, 0) for d in group_data if isinstance(d.get(metric_name), (int, float))]
            
            if not group_values:
                continue
            
            count_below = sum(1 for v in group_values if v < user_value)
            percentile = (count_below / len(group_values)) * 100
            percentiles.append(percentile)
        
        return statistics.mean(percentiles) if percentiles else 50.0
    
    def get_user_reports(self, user_id: str) -> List[AnonymousComparisonReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])


