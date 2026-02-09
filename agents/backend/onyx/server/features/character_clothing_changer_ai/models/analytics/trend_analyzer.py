"""
Trend Analyzer System
=====================
Sistema de análisis de tendencias y patrones
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics


@dataclass
class Trend:
    """Tendencia detectada"""
    metric: str
    direction: str  # 'up', 'down', 'stable'
    strength: float  # 0-1
    period: str  # 'hour', 'day', 'week', 'month'
    data_points: List[Tuple[float, float]]  # (timestamp, value)
    prediction: Optional[float] = None


@dataclass
class Pattern:
    """Patrón detectado"""
    pattern_type: str  # 'seasonal', 'cyclic', 'trend', 'anomaly'
    description: str
    confidence: float  # 0-1
    frequency: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TrendAnalyzer:
    """
    Analizador de tendencias y patrones
    """
    
    def __init__(self):
        self.data_points: Dict[str, List[Tuple[float, float]]] = {}  # metric -> [(timestamp, value)]
        self.trends: Dict[str, Trend] = {}
        self.patterns: List[Pattern] = []
        self.window_size = 100  # Número de puntos para análisis
    
    def record_data_point(self, metric: str, value: float, timestamp: Optional[float] = None):
        """
        Registrar punto de datos
        
        Args:
            metric: Nombre de la métrica
            value: Valor de la métrica
            timestamp: Timestamp (opcional, usa tiempo actual si no se proporciona)
        """
        if timestamp is None:
            timestamp = time.time()
        
        if metric not in self.data_points:
            self.data_points[metric] = []
        
        self.data_points[metric].append((timestamp, value))
        
        # Mantener solo los últimos N puntos
        if len(self.data_points[metric]) > self.window_size:
            self.data_points[metric] = self.data_points[metric][-self.window_size:]
    
    def analyze_trend(
        self,
        metric: str,
        period: str = 'day'
    ) -> Optional[Trend]:
        """
        Analizar tendencia de una métrica
        
        Args:
            metric: Nombre de la métrica
            period: Período de análisis ('hour', 'day', 'week', 'month')
        """
        if metric not in self.data_points or len(self.data_points[metric]) < 2:
            return None
        
        data = self.data_points[metric]
        
        # Calcular dirección de tendencia
        values = [v for _, v in data]
        if len(values) < 2:
            return None
        
        # Regresión lineal simple
        n = len(values)
        x = list(range(n))
        y = values
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            direction = 'stable'
            strength = 0.0
        else:
            slope = numerator / denominator
            
            if slope > 0.01:
                direction = 'up'
            elif slope < -0.01:
                direction = 'down'
            else:
                direction = 'stable'
            
            # Calcular fuerza (normalizada)
            max_value = max(values)
            min_value = min(values)
            if max_value != min_value:
                strength = abs(slope) / ((max_value - min_value) / n)
                strength = min(strength, 1.0)
            else:
                strength = 0.0
        
        # Predicción simple (extrapolación)
        prediction = None
        if len(values) >= 2:
            last_value = values[-1]
            second_last = values[-2]
            prediction = last_value + (last_value - second_last)
        
        trend = Trend(
            metric=metric,
            direction=direction,
            strength=strength,
            period=period,
            data_points=data[-20:],  # Últimos 20 puntos
            prediction=prediction
        )
        
        self.trends[metric] = trend
        return trend
    
    def detect_patterns(self, metric: str) -> List[Pattern]:
        """
        Detectar patrones en una métrica
        
        Args:
            metric: Nombre de la métrica
        """
        if metric not in self.data_points or len(self.data_points[metric]) < 10:
            return []
        
        data = self.data_points[metric]
        values = [v for _, v in data]
        patterns = []
        
        # Detectar patrones estacionales
        seasonal_pattern = self._detect_seasonal_pattern(values)
        if seasonal_pattern:
            patterns.append(seasonal_pattern)
        
        # Detectar patrones cíclicos
        cyclic_pattern = self._detect_cyclic_pattern(values)
        if cyclic_pattern:
            patterns.append(cyclic_pattern)
        
        # Detectar anomalías
        anomalies = self._detect_anomalies(values)
        for anomaly in anomalies:
            patterns.append(anomaly)
        
        self.patterns.extend(patterns)
        return patterns
    
    def _detect_seasonal_pattern(self, values: List[float]) -> Optional[Pattern]:
        """Detectar patrón estacional"""
        if len(values) < 7:
            return None
        
        # Buscar repetición en ventanas de 7 días
        window_size = min(7, len(values) // 2)
        if window_size < 2:
            return None
        
        # Calcular autocorrelación
        correlations = []
        for lag in range(1, window_size + 1):
            if len(values) > lag:
                corr = self._calculate_correlation(
                    values[:-lag],
                    values[lag:]
                )
                correlations.append((lag, corr))
        
        # Encontrar lag con mayor correlación
        if correlations:
            best_lag, best_corr = max(correlations, key=lambda x: x[1])
            if best_corr > 0.7:
                return Pattern(
                    pattern_type='seasonal',
                    description=f'Seasonal pattern with period of {best_lag}',
                    confidence=best_corr,
                    frequency=best_lag,
                    metadata={'lag': best_lag}
                )
        
        return None
    
    def _detect_cyclic_pattern(self, values: List[float]) -> Optional[Pattern]:
        """Detectar patrón cíclico"""
        if len(values) < 5:
            return None
        
        # Calcular variación
        mean = statistics.mean(values)
        variance = statistics.variance(values) if len(values) > 1 else 0
        
        if variance == 0:
            return None
        
        # Detectar ciclos usando FFT (simplificado)
        # En implementación real, usar numpy.fft
        # Por ahora, detectar oscilaciones simples
        
        # Contar cambios de dirección
        direction_changes = 0
        for i in range(1, len(values) - 1):
            if (values[i] > values[i-1] and values[i+1] < values[i]) or \
               (values[i] < values[i-1] and values[i+1] > values[i]):
                direction_changes += 1
        
        if direction_changes > len(values) * 0.3:
            return Pattern(
                pattern_type='cyclic',
                description='Cyclic pattern detected',
                confidence=min(direction_changes / len(values), 1.0),
                metadata={'direction_changes': direction_changes}
            )
        
        return None
    
    def _detect_anomalies(self, values: List[float]) -> List[Pattern]:
        """Detectar anomalías"""
        if len(values) < 3:
            return []
        
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        if stdev == 0:
            return []
        
        anomalies = []
        threshold = 2 * stdev  # 2 desviaciones estándar
        
        for i, value in enumerate(values):
            if abs(value - mean) > threshold:
                anomalies.append(Pattern(
                    pattern_type='anomaly',
                    description=f'Anomaly detected at index {i}',
                    confidence=min(abs(value - mean) / (3 * stdev), 1.0),
                    metadata={
                        'index': i,
                        'value': value,
                        'mean': mean,
                        'stdev': stdev
                    }
                ))
        
        return anomalies
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calcular correlación entre dos series"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        x_var = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        y_var = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
        
        denominator = (x_var * y_var) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_top_trends(self, limit: int = 10) -> List[Trend]:
        """Obtener top tendencias por fuerza"""
        trends = list(self.trends.values())
        trends.sort(key=lambda t: t.strength, reverse=True)
        return trends[:limit]
    
    def get_insights(self) -> Dict[str, Any]:
        """Obtener insights generales"""
        return {
            'total_metrics': len(self.data_points),
            'total_trends': len(self.trends),
            'total_patterns': len(self.patterns),
            'up_trends': len([t for t in self.trends.values() if t.direction == 'up']),
            'down_trends': len([t for t in self.trends.values() if t.direction == 'down']),
            'stable_trends': len([t for t in self.trends.values() if t.direction == 'stable']),
            'anomalies': len([p for p in self.patterns if p.pattern_type == 'anomaly'])
        }


# Instancia global
trend_analyzer = TrendAnalyzer()

