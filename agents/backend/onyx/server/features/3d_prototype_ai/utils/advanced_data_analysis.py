"""
Advanced Data Analysis - Sistema de análisis de datos avanzado
===============================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class AdvancedDataAnalysis:
    """Sistema de análisis de datos avanzado"""
    
    def __init__(self):
        self.datasets: Dict[str, List[Dict[str, Any]]] = {}
        self.analyses: Dict[str, Dict[str, Any]] = {}
    
    def load_dataset(self, dataset_id: str, data: List[Dict[str, Any]]):
        """Carga un dataset"""
        self.datasets[dataset_id] = data
        logger.info(f"Dataset cargado: {dataset_id} - {len(data)} registros")
    
    def analyze_correlation(self, dataset_id: str, variable1: str, variable2: str) -> Dict[str, Any]:
        """Analiza correlación entre variables"""
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset no encontrado: {dataset_id}")
        
        values1 = [d.get(variable1) for d in dataset if variable1 in d]
        values2 = [d.get(variable2) for d in dataset if variable2 in d]
        
        if len(values1) != len(values2) or len(values1) < 2:
            return {
                "correlation": 0.0,
                "confidence": 0.0,
                "message": "Datos insuficientes"
            }
        
        # Calcular correlación de Pearson (simplificado)
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        
        numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(len(values1)))
        denominator1 = sum((v - mean1) ** 2 for v in values1)
        denominator2 = sum((v - mean2) ** 2 for v in values2)
        
        if denominator1 == 0 or denominator2 == 0:
            correlation = 0.0
        else:
            correlation = numerator / ((denominator1 * denominator2) ** 0.5)
        
        return {
            "variable1": variable1,
            "variable2": variable2,
            "correlation": round(correlation, 4),
            "strength": self._interpret_correlation(abs(correlation)),
            "direction": "positive" if correlation > 0 else "negative"
        }
    
    def _interpret_correlation(self, abs_corr: float) -> str:
        """Interpreta fuerza de correlación"""
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def perform_clustering(self, dataset_id: str, features: List[str], k: int = 3) -> Dict[str, Any]:
        """Realiza clustering (K-means simplificado)"""
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset no encontrado: {dataset_id}")
        
        # Clustering simplificado (en producción usaría scikit-learn)
        clusters = {}
        for i, record in enumerate(dataset):
            cluster_id = i % k
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(record)
        
        return {
            "dataset_id": dataset_id,
            "k": k,
            "clusters": {
                str(cluster_id): {
                    "size": len(records),
                    "centroid": self._calculate_centroid(records, features)
                }
                for cluster_id, records in clusters.items()
            },
            "total_records": len(dataset)
        }
    
    def _calculate_centroid(self, records: List[Dict[str, Any]], features: List[str]) -> Dict[str, float]:
        """Calcula centroide de un cluster"""
        centroid = {}
        for feature in features:
            values = [r.get(feature, 0) for r in records if isinstance(r.get(feature), (int, float))]
            if values:
                centroid[feature] = statistics.mean(values)
        return centroid
    
    def detect_outliers(self, dataset_id: str, variable: str) -> List[Dict[str, Any]]:
        """Detecta outliers"""
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset no encontrado: {dataset_id}")
        
        values = [d.get(variable) for d in dataset if isinstance(d.get(variable), (int, float))]
        
        if len(values) < 3:
            return []
        
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        outliers = []
        for i, record in enumerate(dataset):
            value = record.get(variable)
            if isinstance(value, (int, float)):
                z_score = abs((value - mean) / stdev) if stdev > 0 else 0
                if z_score > 2:  # Más de 2 desviaciones estándar
                    outliers.append({
                        "index": i,
                        "value": value,
                        "z_score": round(z_score, 2),
                        "record": record
                    })
        
        return outliers
    
    def generate_statistics(self, dataset_id: str, variable: str) -> Dict[str, Any]:
        """Genera estadísticas descriptivas"""
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset no encontrado: {dataset_id}")
        
        values = [d.get(variable) for d in dataset if isinstance(d.get(variable), (int, float))]
        
        if not values:
            return {"error": "No hay valores numéricos"}
        
        sorted_values = sorted(values)
        
        return {
            "variable": variable,
            "count": len(values),
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "mode": round(statistics.mode(values), 4) if len(set(values)) < len(values) else None,
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "q1": sorted_values[len(sorted_values) // 4] if sorted_values else None,
            "q3": sorted_values[3 * len(sorted_values) // 4] if sorted_values else None
        }




