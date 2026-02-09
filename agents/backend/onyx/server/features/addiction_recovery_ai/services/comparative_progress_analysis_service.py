"""
Servicio de Análisis de Progreso Comparativo - Sistema completo de análisis comparativo
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class ComparativeProgressAnalysisService:
    """Servicio de análisis de progreso comparativo"""
    
    def __init__(self):
        """Inicializa el servicio de análisis comparativo"""
        pass
    
    def compare_periods(
        self,
        user_id: str,
        period1_data: List[Dict],
        period2_data: List[Dict],
        metrics: List[str]
    ) -> Dict:
        """
        Compara dos períodos
        
        Args:
            user_id: ID del usuario
            period1_data: Datos del período 1
            period2_data: Datos del período 2
            metrics: Métricas a comparar
        
        Returns:
            Comparación de períodos
        """
        return {
            "user_id": user_id,
            "comparison_id": f"comparison_{datetime.now().timestamp()}",
            "period1_summary": self._summarize_period(period1_data, metrics),
            "period2_summary": self._summarize_period(period2_data, metrics),
            "comparisons": self._compare_metrics(period1_data, period2_data, metrics),
            "improvements": self._identify_improvements(period1_data, period2_data, metrics),
            "regressions": self._identify_regressions(period1_data, period2_data, metrics),
            "insights": self._generate_comparative_insights(period1_data, period2_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def compare_with_peers(
        self,
        user_id: str,
        user_data: Dict,
        peer_data: List[Dict],
        metrics: List[str]
    ) -> Dict:
        """
        Compara con pares
        
        Args:
            user_id: ID del usuario
            user_data: Datos del usuario
            peer_data: Datos de pares
            metrics: Métricas a comparar
        
        Returns:
            Comparación con pares
        """
        return {
            "user_id": user_id,
            "comparison_type": "peer_comparison",
            "user_metrics": self._extract_metrics(user_data, metrics),
            "peer_averages": self._calculate_peer_averages(peer_data, metrics),
            "percentile_rankings": self._calculate_percentiles(user_data, peer_data, metrics),
            "recommendations": self._generate_peer_recommendations(user_data, peer_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def compare_with_baseline(
        self,
        user_id: str,
        current_data: Dict,
        baseline_data: Dict,
        metrics: List[str]
    ) -> Dict:
        """
        Compara con línea base
        
        Args:
            user_id: ID del usuario
            current_data: Datos actuales
            baseline_data: Datos de línea base
            metrics: Métricas a comparar
        
        Returns:
            Comparación con línea base
        """
        return {
            "user_id": user_id,
            "comparison_type": "baseline_comparison",
            "baseline_metrics": self._extract_metrics(baseline_data, metrics),
            "current_metrics": self._extract_metrics(current_data, metrics),
            "changes": self._calculate_changes(baseline_data, current_data, metrics),
            "improvement_percentage": self._calculate_improvement(baseline_data, current_data, metrics),
            "generated_at": datetime.now().isoformat()
        }
    
    def _summarize_period(self, data: List[Dict], metrics: List[str]) -> Dict:
        """Resume período"""
        summary = {}
        
        for metric in metrics:
            values = [d.get(metric, 0) for d in data if metric in d]
            if values:
                summary[metric] = {
                    "average": round(statistics.mean(values), 2),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    def _compare_metrics(self, period1: List[Dict], period2: List[Dict], metrics: List[str]) -> Dict:
        """Compara métricas"""
        comparisons = {}
        
        for metric in metrics:
            values1 = [d.get(metric, 0) for d in period1 if metric in d]
            values2 = [d.get(metric, 0) for d in period2 if metric in d]
            
            if values1 and values2:
                avg1 = statistics.mean(values1)
                avg2 = statistics.mean(values2)
                
                comparisons[metric] = {
                    "period1_avg": round(avg1, 2),
                    "period2_avg": round(avg2, 2),
                    "change": round(avg2 - avg1, 2),
                    "change_percentage": round(((avg2 - avg1) / avg1 * 100) if avg1 > 0 else 0, 2),
                    "direction": "improved" if avg2 > avg1 else "declined" if avg2 < avg1 else "stable"
                }
        
        return comparisons
    
    def _identify_improvements(self, period1: List[Dict], period2: List[Dict], metrics: List[str]) -> List[str]:
        """Identifica mejoras"""
        improvements = []
        
        comparisons = self._compare_metrics(period1, period2, metrics)
        
        for metric, comp in comparisons.items():
            if comp.get("direction") == "improved":
                improvements.append(f"Mejora en {metric}: {comp.get('change_percentage')}%")
        
        return improvements
    
    def _identify_regressions(self, period1: List[Dict], period2: List[Dict], metrics: List[str]) -> List[str]:
        """Identifica regresiones"""
        regressions = []
        
        comparisons = self._compare_metrics(period1, period2, metrics)
        
        for metric, comp in comparisons.items():
            if comp.get("direction") == "declined":
                regressions.append(f"Regresión en {metric}: {comp.get('change_percentage')}%")
        
        return regressions
    
    def _generate_comparative_insights(self, period1: List[Dict], period2: List[Dict]) -> List[str]:
        """Genera insights comparativos"""
        insights = []
        
        if len(period2) > len(period1):
            insights.append("Mayor actividad registrada en el período reciente")
        
        return insights
    
    def _extract_metrics(self, data: Dict, metrics: List[str]) -> Dict:
        """Extrae métricas"""
        return {metric: data.get(metric, 0) for metric in metrics}
    
    def _calculate_peer_averages(self, peer_data: List[Dict], metrics: List[str]) -> Dict:
        """Calcula promedios de pares"""
        averages = {}
        
        for metric in metrics:
            values = [p.get(metric, 0) for p in peer_data if metric in p]
            if values:
                averages[metric] = round(statistics.mean(values), 2)
        
        return averages
    
    def _calculate_percentiles(self, user_data: Dict, peer_data: List[Dict], metrics: List[str]) -> Dict:
        """Calcula percentiles"""
        percentiles = {}
        
        for metric in metrics:
            user_value = user_data.get(metric, 0)
            peer_values = [p.get(metric, 0) for p in peer_data if metric in p]
            
            if peer_values:
                below_count = sum(1 for v in peer_values if v < user_value)
                percentile = (below_count / len(peer_values)) * 100
                percentiles[metric] = round(percentile, 2)
        
        return percentiles
    
    def _generate_peer_recommendations(self, user_data: Dict, peer_data: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en comparación con pares"""
        recommendations = []
        
        # Lógica simplificada
        recommendations.append("Tu progreso está en línea con otros usuarios en recuperación")
        
        return recommendations
    
    def _calculate_changes(self, baseline: Dict, current: Dict, metrics: List[str]) -> Dict:
        """Calcula cambios"""
        changes = {}
        
        for metric in metrics:
            baseline_value = baseline.get(metric, 0)
            current_value = current.get(metric, 0)
            
            changes[metric] = {
                "baseline": baseline_value,
                "current": current_value,
                "change": round(current_value - baseline_value, 2),
                "change_percentage": round(((current_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0, 2)
            }
        
        return changes
    
    def _calculate_improvement(self, baseline: Dict, current: Dict, metrics: List[str]) -> float:
        """Calcula porcentaje de mejora"""
        improvements = []
        
        for metric in metrics:
            baseline_value = baseline.get(metric, 0)
            current_value = current.get(metric, 0)
            
            if baseline_value > 0:
                improvement = ((current_value - baseline_value) / baseline_value) * 100
                improvements.append(improvement)
        
        return round(statistics.mean(improvements), 2) if improvements else 0.0

