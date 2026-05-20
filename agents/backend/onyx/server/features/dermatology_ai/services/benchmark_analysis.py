"""
Sistema de análisis de progreso con benchmarks
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class Benchmark:
    """Benchmark de métrica"""
    metric_name: str
    user_value: float
    benchmark_value: float
    percentile: float
    status: str  # "excellent", "good", "average", "below_average"
    improvement_needed: float
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric_name": self.metric_name,
            "user_value": self.user_value,
            "benchmark_value": self.benchmark_value,
            "percentile": self.percentile,
            "status": self.status,
            "improvement_needed": self.improvement_needed
        }


@dataclass
class BenchmarkReport:
    """Reporte de benchmarks"""
    user_id: str
    benchmarks: List[Benchmark]
    overall_percentile: float
    recommendations: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "overall_percentile": self.overall_percentile,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class BenchmarkAnalysis:
    """Sistema de análisis con benchmarks"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.benchmark_data: Dict[str, List[float]] = {}  # metric -> [values from all users]
        self.user_scores: Dict[str, Dict] = {}  # user_id -> {metric: value}
    
    def add_user_score(self, user_id: str, metric_name: str, value: float):
        """Agrega score de usuario"""
        if metric_name not in self.benchmark_data:
            self.benchmark_data[metric_name] = []
        
        self.benchmark_data[metric_name].append(value)
        
        if user_id not in self.user_scores:
            self.user_scores[user_id] = {}
        
        self.user_scores[user_id][metric_name] = value
    
    def generate_benchmark_report(self, user_id: str) -> BenchmarkReport:
        """Genera reporte de benchmarks"""
        user_scores = self.user_scores.get(user_id, {})
        
        if not user_scores:
            return BenchmarkReport(
                user_id=user_id,
                benchmarks=[],
                overall_percentile=0.0,
                recommendations=["No hay datos suficientes para generar benchmarks"]
            )
        
        benchmarks = []
        
        for metric_name, user_value in user_scores.items():
            metric_data = self.benchmark_data.get(metric_name, [])
            
            if not metric_data:
                continue
            
            # Calcular benchmark (promedio)
            benchmark_value = statistics.mean(metric_data)
            
            # Calcular percentil
            percentile = self._calculate_percentile(user_value, metric_data)
            
            # Determinar status
            if percentile >= 90:
                status = "excellent"
            elif percentile >= 70:
                status = "good"
            elif percentile >= 50:
                status = "average"
            else:
                status = "below_average"
            
            # Calcular mejora necesaria
            improvement_needed = max(0, benchmark_value - user_value)
            
            benchmark = Benchmark(
                metric_name=metric_name,
                user_value=user_value,
                benchmark_value=benchmark_value,
                percentile=percentile,
                status=status,
                improvement_needed=improvement_needed
            )
            benchmarks.append(benchmark)
        
        # Percentil general
        overall_percentile = statistics.mean([b.percentile for b in benchmarks]) if benchmarks else 0.0
        
        # Recomendaciones
        recommendations = self._generate_recommendations(benchmarks)
        
        return BenchmarkReport(
            user_id=user_id,
            benchmarks=benchmarks,
            overall_percentile=overall_percentile,
            recommendations=recommendations
        )
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calcula percentil"""
        if not values:
            return 50.0
        
        sorted_values = sorted(values)
        count_below = sum(1 for v in sorted_values if v < value)
        percentile = (count_below / len(sorted_values)) * 100
        
        return float(percentile)
    
    def _generate_recommendations(self, benchmarks: List[Benchmark]) -> List[str]:
        """Genera recomendaciones basadas en benchmarks"""
        recommendations = []
        
        below_average = [b for b in benchmarks if b.status == "below_average"]
        
        if below_average:
            recommendations.append(f"Tienes {len(below_average)} métricas por debajo del promedio")
            recommendations.append("Enfócate en mejorar estas áreas primero")
        
        excellent = [b for b in benchmarks if b.status == "excellent"]
        if excellent:
            recommendations.append(f"¡Excelente! {len(excellent)} métricas están en el top 10%")
        
        # Recomendación específica para la métrica más baja
        if benchmarks:
            lowest = min(benchmarks, key=lambda x: x.percentile)
            recommendations.append(
                f"Prioriza mejorar {lowest.metric_name} "
                f"(necesitas {lowest.improvement_needed:.1f} puntos para alcanzar el promedio)"
            )
        
        return recommendations






