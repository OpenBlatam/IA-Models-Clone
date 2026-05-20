"""
Sistema de comparación avanzada
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import statistics


@dataclass
class ComparisonResult:
    """Resultado de comparación"""
    metric: str
    value1: float
    value2: float
    difference: float
    percentage_change: float
    improvement: bool  # True si mejoró
    significance: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric": self.metric,
            "value1": self.value1,
            "value2": self.value2,
            "difference": self.difference,
            "percentage_change": self.percentage_change,
            "improvement": self.improvement,
            "significance": self.significance
        }


class AdvancedComparison:
    """Sistema de comparación avanzada"""
    
    def __init__(self):
        """Inicializa el sistema"""
        pass
    
    def compare_analyses(self, analysis1: Dict, analysis2: Dict) -> Dict:
        """
        Compara dos análisis
        
        Args:
            analysis1: Primer análisis
            analysis2: Segundo análisis
            
        Returns:
            Diccionario con comparación
        """
        comparison = {
            "analysis1_id": analysis1.get("id", "unknown"),
            "analysis2_id": analysis2.get("id", "unknown"),
            "comparison_date": datetime.now().isoformat(),
            "metrics": [],
            "overall_summary": {}
        }
        
        # Comparar métricas de calidad
        scores1 = analysis1.get("quality_scores", {})
        scores2 = analysis2.get("quality_scores", {})
        
        for metric in scores1.keys():
            if metric in scores2:
                value1 = scores1[metric]
                value2 = scores2[metric]
                
                result = self._compare_metric(metric, value1, value2)
                comparison["metrics"].append(result.to_dict())
        
        # Resumen general
        comparison["overall_summary"] = self._generate_summary(comparison["metrics"])
        
        return comparison
    
    def _compare_metric(self, metric: str, value1: float, value2: float) -> ComparisonResult:
        """Compara una métrica"""
        difference = value2 - value1
        percentage_change = ((value2 - value1) / value1 * 100) if value1 > 0 else 0
        improvement = difference > 0
        
        # Determinar significancia
        abs_change = abs(percentage_change)
        if abs_change >= 10:
            significance = "high"
        elif abs_change >= 5:
            significance = "medium"
        else:
            significance = "low"
        
        return ComparisonResult(
            metric=metric,
            value1=value1,
            value2=value2,
            difference=difference,
            percentage_change=percentage_change,
            improvement=improvement,
            significance=significance
        )
    
    def _generate_summary(self, metrics: List[Dict]) -> Dict:
        """Genera resumen de comparación"""
        improvements = sum(1 for m in metrics if m["improvement"])
        total_metrics = len(metrics)
        
        high_significance = sum(1 for m in metrics if m["significance"] == "high")
        
        avg_change = statistics.mean([abs(m["percentage_change"]) for m in metrics]) if metrics else 0
        
        return {
            "total_metrics": total_metrics,
            "improved_metrics": improvements,
            "worsened_metrics": total_metrics - improvements,
            "improvement_rate": (improvements / total_metrics * 100) if total_metrics > 0 else 0,
            "high_significance_changes": high_significance,
            "average_change": avg_change,
            "overall_trend": "improving" if improvements > total_metrics / 2 else "declining"
        }
    
    def compare_multiple(self, analyses: List[Dict]) -> Dict:
        """
        Compara múltiples análisis
        
        Args:
            analyses: Lista de análisis
            
        Returns:
            Comparación múltiple
        """
        if len(analyses) < 2:
            return {"error": "Se requieren al menos 2 análisis"}
        
        # Comparar todos con el primero
        base_analysis = analyses[0]
        comparisons = []
        
        for i, analysis in enumerate(analyses[1:], 1):
            comparison = self.compare_analyses(base_analysis, analysis)
            comparison["comparison_index"] = i
            comparisons.append(comparison)
        
        # Análisis de tendencias
        trends = self._analyze_trends(analyses)
        
        return {
            "base_analysis_id": base_analysis.get("id", "unknown"),
            "total_comparisons": len(comparisons),
            "comparisons": comparisons,
            "trends": trends
        }
    
    def _analyze_trends(self, analyses: List[Dict]) -> Dict:
        """Analiza tendencias en múltiples análisis"""
        trends = {}
        
        # Extraer scores a lo largo del tiempo
        overall_scores = []
        for analysis in analyses:
            score = analysis.get("quality_scores", {}).get("overall_score", 0)
            overall_scores.append(score)
        
        if len(overall_scores) >= 2:
            # Calcular tendencia
            first_score = overall_scores[0]
            last_score = overall_scores[-1]
            change = last_score - first_score
            
            trends["overall_score"] = {
                "first": first_score,
                "last": last_score,
                "change": change,
                "percentage_change": (change / first_score * 100) if first_score > 0 else 0,
                "trend": "improving" if change > 0 else "declining" if change < 0 else "stable"
            }
        
        return trends






