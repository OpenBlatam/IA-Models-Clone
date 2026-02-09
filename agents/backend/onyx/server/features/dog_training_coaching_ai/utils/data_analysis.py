"""
Data Analysis Utilities
=======================
Utilidades para análisis de datos.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from statistics import mean, median, mode, stdev, variance
from collections import Counter

from .logger import get_logger

logger = get_logger(__name__)


class DataAnalyzer:
    """Analizador de datos."""
    
    def __init__(self, data: List[Any]):
        """
        Inicializar analizador.
        
        Args:
            data: Datos a analizar
        """
        self.data = data
    
    def describe(self) -> Dict[str, Any]:
        """Obtener descripción estadística de los datos."""
        if not self.data:
            return {"error": "No data to analyze"}
        
        numeric_data = [x for x in self.data if isinstance(x, (int, float))]
        
        if not numeric_data:
            return {
                "count": len(self.data),
                "type": "non-numeric",
                "unique_values": len(set(self.data))
            }
        
        return {
            "count": len(numeric_data),
            "mean": mean(numeric_data),
            "median": median(numeric_data),
            "mode": mode(numeric_data) if len(set(numeric_data)) < len(numeric_data) else None,
            "min": min(numeric_data),
            "max": max(numeric_data),
            "std_dev": stdev(numeric_data) if len(numeric_data) > 1 else 0,
            "variance": variance(numeric_data) if len(numeric_data) > 1 else 0,
            "range": max(numeric_data) - min(numeric_data)
        }
    
    def frequency_distribution(self) -> Dict[Any, int]:
        """Obtener distribución de frecuencias."""
        return dict(Counter(self.data))
    
    def filter(self, predicate: Callable[[Any], bool]) -> List[Any]:
        """
        Filtrar datos.
        
        Args:
            predicate: Función de filtrado
            
        Returns:
            Datos filtrados
        """
        return [x for x in self.data if predicate(x)]
    
    def group_by(self, key_func: Callable[[Any], Any]) -> Dict[Any, List[Any]]:
        """
        Agrupar datos por clave.
        
        Args:
            key_func: Función para extraer clave
            
        Returns:
            Datos agrupados
        """
        grouped = {}
        for item in self.data:
            key = key_func(item)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        return grouped
    
    def aggregate(
        self,
        group_key: Callable[[Any], Any],
        agg_func: Callable[[List[Any]], Any]
    ) -> Dict[Any, Any]:
        """
        Agregar datos.
        
        Args:
            group_key: Función para clave de agrupación
            agg_func: Función de agregación
            
        Returns:
            Datos agregados
        """
        grouped = self.group_by(group_key)
        return {key: agg_func(values) for key, values in grouped.items()}


def analyze_trends(data: List[Dict[str, Any]], value_key: str, time_key: str) -> Dict[str, Any]:
    """
    Analizar tendencias en datos temporales.
    
    Args:
        data: Datos con timestamps
        value_key: Clave del valor a analizar
        time_key: Clave del timestamp
        
    Returns:
        Análisis de tendencias
    """
    if not data:
        return {"error": "No data provided"}
    
    sorted_data = sorted(data, key=lambda x: x.get(time_key, ""))
    values = [x.get(value_key, 0) for x in sorted_data if isinstance(x.get(value_key), (int, float))]
    
    if len(values) < 2:
        return {"error": "Insufficient data for trend analysis"}
    
    # Calcular tendencia (simple regresión lineal)
    n = len(values)
    x = list(range(n))
    x_mean = mean(x)
    y_mean = mean(values)
    
    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean
    
    # Determinar dirección de tendencia
    if slope > 0.1:
        trend = "increasing"
    elif slope < -0.1:
        trend = "decreasing"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "slope": slope,
        "intercept": intercept,
        "first_value": values[0],
        "last_value": values[-1],
        "change": values[-1] - values[0],
        "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
        "data_points": n
    }


def detect_anomalies(
    data: List[float],
    threshold: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Detectar anomalías en datos.
    
    Args:
        data: Datos numéricos
        threshold: Threshold de desviación estándar
        
    Returns:
        Lista de anomalías detectadas
    """
    if len(data) < 3:
        return []
    
    mean_val = mean(data)
    std_val = stdev(data)
    
    anomalies = []
    for i, value in enumerate(data):
        z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
        if z_score > threshold:
            anomalies.append({
                "index": i,
                "value": value,
                "z_score": z_score,
                "deviation": value - mean_val
            })
    
    return anomalies


def calculate_correlation(x: List[float], y: List[float]) -> Dict[str, Any]:
    """
    Calcular correlación entre dos variables.
    
    Args:
        x: Primera variable
        y: Segunda variable
        
    Returns:
        Estadísticas de correlación
    """
    if len(x) != len(y) or len(x) < 2:
        return {"error": "Invalid data for correlation"}
    
    n = len(x)
    x_mean = mean(x)
    y_mean = mean(y)
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    x_std = stdev(x) if len(x) > 1 else 1
    y_std = stdev(y) if len(y) > 1 else 1
    
    correlation = numerator / (n * x_std * y_std) if (x_std * y_std) != 0 else 0
    
    # Interpretación
    if abs(correlation) > 0.7:
        strength = "strong"
    elif abs(correlation) > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    
    direction = "positive" if correlation > 0 else "negative"
    
    return {
        "correlation": correlation,
        "strength": strength,
        "direction": direction,
        "n": n
    }



