"""
Prometheus Metrics - Sistema de métricas Prometheus
====================================================
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Sistema de métricas compatible con Prometheus"""
    
    def __init__(self):
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)
        self.summaries: Dict[str, list] = defaultdict(list)
        self.labels: Dict[str, Dict[str, str]] = {}
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Incrementa un contador"""
        key = self._get_key(name, labels)
        self.counters[key] += value
        if labels:
            self.labels[key] = labels
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Establece un gauge"""
        key = self._get_key(name, labels)
        self.gauges[key] = value
        if labels:
            self.labels[key] = labels
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observa un valor en un histograma"""
        key = self._get_key(name, labels)
        self.histograms[key].append(value)
        # Mantener solo últimas 1000 observaciones
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        if labels:
            self.labels[key] = labels
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observa un valor en un summary"""
        key = self._get_key(name, labels)
        self.summaries[key].append(value)
        if len(self.summaries[key]) > 1000:
            self.summaries[key] = self.summaries[key][-1000:]
        if labels:
            self.labels[key] = labels
    
    def _get_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Genera una clave única para métrica con labels"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def get_metrics_prometheus_format(self) -> str:
        """Obtiene métricas en formato Prometheus"""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # Histograms
        for key, values in self.histograms.items():
            if values:
                metric_name = key.split('{')[0]
                lines.append(f"# TYPE {metric_name} histogram")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_bucket{{le=\"+Inf\"}} {len(values)}")
        
        # Summaries
        for key, values in self.summaries.items():
            if values:
                metric_name = key.split('{')[0]
                lines.append(f"# TYPE {metric_name} summary")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                sorted_values = sorted(values)
                if sorted_values:
                    lines.append(f"{key}{{quantile=\"0.5\"}} {sorted_values[len(sorted_values)//2]}")
                    lines.append(f"{key}{{quantile=\"0.95\"}} {sorted_values[int(len(sorted_values)*0.95)]}")
                    lines.append(f"{key}{{quantile=\"0.99\"}} {sorted_values[int(len(sorted_values)*0.99)]}")
        
        return "\n".join(lines)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Obtiene métricas como diccionario"""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: {
                "count": len(v),
                "sum": sum(v),
                "min": min(v) if v else 0,
                "max": max(v) if v else 0,
                "avg": sum(v) / len(v) if v else 0
            } for k, v in self.histograms.items()},
            "summaries": {k: {
                "count": len(v),
                "sum": sum(v),
                "avg": sum(v) / len(v) if v else 0
            } for k, v in self.summaries.items()}
        }




