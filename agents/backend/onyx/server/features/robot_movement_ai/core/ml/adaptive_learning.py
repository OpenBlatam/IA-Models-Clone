"""
Adaptive Learning System
========================

Sistema de aprendizaje adaptativo para optimización continua.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class LearningMetric:
    """Métrica de aprendizaje."""
    metric_id: str
    name: str
    value: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Patrón de aprendizaje."""
    pattern_id: str
    name: str
    pattern_type: str
    confidence: float
    parameters: Dict[str, Any]
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveLearningSystem:
    """
    Sistema de aprendizaje adaptativo.
    
    Aprende y adapta parámetros basándose en métricas históricas.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        learning_rate: float = 0.01,
        min_samples: int = 100
    ):
        """
        Inicializar sistema de aprendizaje adaptativo.
        
        Args:
            window_size: Tamaño de ventana para métricas
            learning_rate: Tasa de aprendizaje
            min_samples: Mínimo de muestras para aprendizaje
        """
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        
        self.metrics_history: deque = deque(maxlen=window_size)
        self.patterns: List[LearningPattern] = []
        self.adaptive_parameters: Dict[str, float] = {}
        self.performance_trends: Dict[str, List[float]] = {}
    
    def record_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Registrar métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            metadata: Metadata adicional
            
        Returns:
            ID de la métrica
        """
        import uuid
        metric_id = str(uuid.uuid4())
        
        metric = LearningMetric(
            metric_id=metric_id,
            name=name,
            value=value,
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metric)
        
        # Actualizar tendencias
        if name not in self.performance_trends:
            self.performance_trends[name] = []
        self.performance_trends[name].append(value)
        if len(self.performance_trends[name]) > self.window_size:
            self.performance_trends[name] = self.performance_trends[name][-self.window_size:]
        
        return metric_id
    
    def detect_patterns(self) -> List[LearningPattern]:
        """
        Detectar patrones en métricas.
        
        Returns:
            Lista de patrones detectados
        """
        if len(self.metrics_history) < self.min_samples:
            return []
        
        patterns = []
        
        # Detectar tendencias
        for metric_name, values in self.performance_trends.items():
            if len(values) < self.min_samples:
                continue
            
            # Calcular tendencia
            trend = self._calculate_trend(values)
            
            if abs(trend) > 0.1:  # Umbral de significancia
                pattern = LearningPattern(
                    pattern_id=f"pattern_{len(patterns)}",
                    name=f"{metric_name}_trend",
                    pattern_type="trend",
                    confidence=min(abs(trend), 1.0),
                    parameters={
                        "trend": trend,
                        "metric_name": metric_name,
                        "direction": "increasing" if trend > 0 else "decreasing"
                    }
                )
                patterns.append(pattern)
        
        # Detectar ciclos
        cycles = self._detect_cycles()
        patterns.extend(cycles)
        
        # Detectar anomalías
        anomalies = self._detect_anomalies()
        patterns.extend(anomalies)
        
        self.patterns = patterns
        return patterns
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calcular tendencia de valores."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Regresión lineal simple
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalizar por rango
        value_range = max(values) - min(values)
        if value_range > 0:
            normalized_trend = slope / value_range
        else:
            normalized_trend = 0.0
        
        return normalized_trend
    
    def _detect_cycles(self) -> List[LearningPattern]:
        """
        Detectar ciclos en métricas usando análisis de frecuencia FFT.
        
        Analiza componentes de frecuencia en las métricas históricas
        para identificar patrones cíclicos y oscilatorios.
        
        Returns:
            Lista de patrones de ciclo detectados con información de frecuencia
        """
        cycles: List[LearningPattern] = []
        
        for metric_name, values in self.performance_trends.items():
            if len(values) < 50:
                continue
            
            try:
                values_array = np.array(values, dtype=np.float64)
                
                if np.any(np.isnan(values_array)) or np.any(np.isinf(values_array)):
                    logger.warning(f"Valores inválidos en métrica {metric_name}, omitiendo")
                    continue
                
                fft_values = np.fft.fft(values_array)
                frequencies = np.fft.fftfreq(len(values_array))
                
                power = np.abs(fft_values)
                positive_mask = frequencies > 0
                positive_frequencies = frequencies[positive_mask]
                positive_power = power[positive_mask]
                
                if len(positive_power) == 0:
                    continue
                
                dominant_freq_idx = np.argmax(positive_power)
                dominant_frequency = positive_frequencies[dominant_freq_idx]
                dominant_power = positive_power[dominant_freq_idx]
                
                mean_power = np.mean(positive_power)
                power_threshold = mean_power * 2.0
                
                if dominant_power > power_threshold:
                    cycle_length = len(values_array) / abs(dominant_frequency) if dominant_frequency != 0 else 0.0
                    confidence = min(dominant_power / np.max(positive_power), 1.0)
                    
                    pattern = LearningPattern(
                        pattern_id=f"cycle_{len(cycles)}",
                        name=f"{metric_name}_cycle",
                        pattern_type="cycle",
                        confidence=float(confidence),
                        parameters={
                            "cycle_length": float(cycle_length),
                            "metric_name": metric_name,
                            "frequency": float(dominant_frequency),
                            "power": float(dominant_power),
                            "mean_power": float(mean_power)
                        }
                    )
                    cycles.append(pattern)
            except Exception as e:
                logger.error(f"Error detectando ciclos en {metric_name}: {e}")
                continue
        
        return cycles
    
    def _detect_anomalies(self) -> List[LearningPattern]:
        """Detectar anomalías en métricas."""
        anomalies = []
        
        for metric_name, values in self.performance_trends.items():
            if len(values) < 20:
                continue
            
            mean = np.mean(values)
            std = np.std(values)
            
            if std > 0:
                # Detectar valores fuera de 3 desviaciones estándar
                z_scores = np.abs((np.array(values) - mean) / std)
                anomaly_indices = np.where(z_scores > 3)[0]
                
                if len(anomaly_indices) > 0:
                    pattern = LearningPattern(
                        pattern_id=f"anomaly_{len(anomalies)}",
                        name=f"{metric_name}_anomaly",
                        pattern_type="anomaly",
                        confidence=min(len(anomaly_indices) / len(values), 1.0),
                        parameters={
                            "metric_name": metric_name,
                            "anomaly_count": len(anomaly_indices),
                            "mean": mean,
                            "std": std
                        }
                    )
                    anomalies.append(pattern)
        
        return anomalies
    
    def adapt_parameters(
        self,
        parameter_name: str,
        current_value: float,
        performance_metric: float,
        target_metric: float = 1.0
    ) -> float:
        """
        Adaptar parámetro basándose en rendimiento.
        
        Args:
            parameter_name: Nombre del parámetro
            current_value: Valor actual
            performance_metric: Métrica de rendimiento actual
            target_metric: Métrica objetivo
            
        Returns:
            Nuevo valor del parámetro
        """
        error = target_metric - performance_metric
        
        # Ajuste adaptativo
        adjustment = self.learning_rate * error * current_value
        
        new_value = current_value + adjustment
        
        # Guardar parámetro adaptativo
        self.adaptive_parameters[parameter_name] = new_value
        
        return new_value
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones basadas en patrones.
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar patrones
        for pattern in self.patterns:
            if pattern.pattern_type == "trend":
                if pattern.parameters["direction"] == "decreasing":
                    recommendations.append({
                        "type": "optimization",
                        "message": f"Metric {pattern.parameters['metric_name']} is decreasing. Consider optimization.",
                        "confidence": pattern.confidence,
                        "pattern": pattern.name
                    })
            elif pattern.pattern_type == "anomaly":
                recommendations.append({
                    "type": "investigation",
                    "message": f"Anomalies detected in {pattern.parameters['metric_name']}. Investigate root cause.",
                    "confidence": pattern.confidence,
                    "pattern": pattern.name
                })
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema."""
        pattern_counts = {}
        for pattern in self.patterns:
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
        
        return {
            "total_metrics": len(self.metrics_history),
            "total_patterns": len(self.patterns),
            "pattern_counts": pattern_counts,
            "adaptive_parameters": len(self.adaptive_parameters),
            "tracked_metrics": len(self.performance_trends)
        }


# Instancia global
_adaptive_learning_system: Optional[AdaptiveLearningSystem] = None


def get_adaptive_learning_system(
    window_size: int = 1000,
    learning_rate: float = 0.01,
    min_samples: int = 100
) -> AdaptiveLearningSystem:
    """Obtener instancia global del sistema de aprendizaje adaptativo."""
    global _adaptive_learning_system
    if _adaptive_learning_system is None:
        _adaptive_learning_system = AdaptiveLearningSystem(
            window_size=window_size,
            learning_rate=learning_rate,
            min_samples=min_samples
        )
    return _adaptive_learning_system


