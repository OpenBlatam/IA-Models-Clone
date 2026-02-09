"""
Model Monitoring Module
========================

Sistema profesional de monitoreo para modelos en producción.
Incluye drift detection, performance monitoring, y alertas.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Métricas de modelo."""
    timestamp: datetime
    accuracy: float = 0.0
    latency: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    predictions_count: int = 0


class DataDriftDetector:
    """
    Detector de drift de datos.
    
    Detecta cambios en la distribución de datos de entrada.
    """
    
    def __init__(
        self,
        reference_data: np.ndarray,
        threshold: float = 0.05
    ):
        """
        Inicializar detector.
        
        Args:
            reference_data: Datos de referencia
            threshold: Umbral para detección
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_stats(reference_data)
        logger.info("DataDriftDetector initialized")
    
    def _compute_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calcular estadísticas de datos."""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, Any]:
        """
        Detectar drift en nuevos datos.
        
        Args:
            new_data: Nuevos datos
            
        Returns:
            Dict con resultado de detección
        """
        new_stats = self._compute_stats(new_data)
        
        # Calcular distancia entre distribuciones
        mean_diff = np.mean(np.abs(new_stats['mean'] - self.reference_stats['mean']))
        std_diff = np.mean(np.abs(new_stats['std'] - self.reference_stats['std']))
        
        drift_score = (mean_diff + std_diff) / 2.0
        has_drift = drift_score > self.threshold
        
        return {
            'has_drift': has_drift,
            'drift_score': float(drift_score),
            'threshold': self.threshold,
            'mean_diff': float(mean_diff),
            'std_diff': float(std_diff)
        }


class PerformanceMonitor:
    """
    Monitor de performance de modelos.
    
    Rastrea métricas de performance en tiempo real.
    """
    
    def __init__(self, model_name: str = "model"):
        """
        Inicializar monitor.
        
        Args:
            model_name: Nombre del modelo
        """
        self.model_name = model_name
        self.metrics_history: List[ModelMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        logger.info(f"PerformanceMonitor initialized for {model_name}")
    
    def record_prediction(
        self,
        latency: float,
        is_correct: Optional[bool] = None,
        memory_usage: Optional[float] = None
    ):
        """
        Registrar predicción.
        
        Args:
            latency: Latencia en segundos
            is_correct: Si la predicción fue correcta (opcional)
            memory_usage: Uso de memoria en MB (opcional)
        """
        if not self.metrics_history:
            metrics = ModelMetrics(
                timestamp=datetime.now(),
                latency=latency,
                accuracy=1.0 if is_correct else 0.0,
                memory_usage=memory_usage or 0.0,
                predictions_count=1
            )
        else:
            last_metrics = self.metrics_history[-1]
            total_correct = (last_metrics.accuracy * last_metrics.predictions_count) + (1 if is_correct else 0)
            new_count = last_metrics.predictions_count + 1
            
            metrics = ModelMetrics(
                timestamp=datetime.now(),
                latency=latency,
                accuracy=total_correct / new_count if new_count > 0 else 0.0,
                memory_usage=memory_usage or last_metrics.memory_usage,
                predictions_count=new_count
            )
        
        self.metrics_history.append(metrics)
        
        # Verificar alertas
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: ModelMetrics):
        """Verificar condiciones de alerta."""
        # Alerta de latencia alta
        if metrics.latency > 1.0:  # Más de 1 segundo
            self.alerts.append({
                'timestamp': metrics.timestamp,
                'type': 'high_latency',
                'value': metrics.latency,
                'message': f"High latency detected: {metrics.latency:.2f}s"
            })
        
        # Alerta de accuracy baja
        if metrics.accuracy < 0.5 and metrics.predictions_count > 100:
            self.alerts.append({
                'timestamp': metrics.timestamp,
                'type': 'low_accuracy',
                'value': metrics.accuracy,
                'message': f"Low accuracy detected: {metrics.accuracy:.2%}"
            })
        
        # Alerta de uso de memoria alto
        if metrics.memory_usage > 8000:  # Más de 8GB
            self.alerts.append({
                'timestamp': metrics.timestamp,
                'type': 'high_memory',
                'value': metrics.memory_usage,
                'message': f"High memory usage: {metrics.memory_usage:.2f} MB"
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de métricas.
        
        Returns:
            Dict con resumen
        """
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-100:]  # Últimas 100 predicciones
        
        return {
            "model_name": self.model_name,
            "total_predictions": len(self.metrics_history),
            "recent_accuracy": np.mean([m.accuracy for m in recent_metrics]),
            "avg_latency": np.mean([m.latency for m in recent_metrics]),
            "p95_latency": np.percentile([m.latency for m in recent_metrics], 95),
            "avg_memory": np.mean([m.memory_usage for m in recent_metrics]),
            "alerts_count": len(self.alerts),
            "recent_alerts": self.alerts[-10:] if self.alerts else []
        }


class ModelRegistry:
    """
    Registro de modelos para versionado y gestión.
    
    Permite registrar, versionar y gestionar múltiples modelos.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Inicializar registro.
        
        Args:
            registry_path: Ruta del registro
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, Any]] = {}
        logger.info(f"ModelRegistry initialized at {registry_path}")
    
    def register_model(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Registrar modelo.
        
        Args:
            model: Modelo PyTorch
            model_name: Nombre del modelo
            version: Versión
            metadata: Metadatos adicionales
            
        Returns:
            ID del modelo registrado
        """
        model_id = f"{model_name}_v{version}"
        
        # Guardar modelo
        model_path = self.registry_path / f"{model_id}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'version': version,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        # Registrar en memoria
        self.models[model_id] = {
            'name': model_name,
            'version': version,
            'path': str(model_path),
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat()
        }
        
        logger.info(f"Model registered: {model_id}")
        return model_id
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Listar modelos registrados.
        
        Returns:
            Lista de modelos
        """
        return list(self.models.values())
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información de un modelo.
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Información del modelo o None
        """
        return self.models.get(model_id)
    
    def load_model(self, model_id: str, model_class: type) -> Optional[nn.Module]:
        """
        Cargar modelo del registro.
        
        Args:
            model_id: ID del modelo
            model_class: Clase del modelo
            
        Returns:
            Modelo cargado o None
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return None
        
        model_path = self.models[model_id]['path']
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model {model_id} loaded from registry")
        return model

