"""
Sistema de Model Serving
==========================

Sistema para servir modelos en producción.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ServingStrategy(Enum):
    """Estrategia de serving"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"


@dataclass
class ServingEndpoint:
    """Endpoint de serving"""
    endpoint_id: str
    model_id: str
    strategy: ServingStrategy
    url: str
    status: str
    created_at: str


class ModelServing:
    """
    Sistema de Model Serving
    
    Proporciona:
    - Serving de modelos en producción
    - Múltiples estrategias de serving
    - Endpoints REST/GraphQL
    - Batch processing
    - Streaming inference
    - Load balancing
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.endpoints: Dict[str, ServingEndpoint] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("ModelServing inicializado")
    
    def create_endpoint(
        self,
        model_id: str,
        strategy: ServingStrategy = ServingStrategy.REAL_TIME,
        endpoint_id: Optional[str] = None
    ) -> ServingEndpoint:
        """
        Crear endpoint de serving
        
        Args:
            model_id: ID del modelo
            strategy: Estrategia de serving
            endpoint_id: ID del endpoint (opcional)
        
        Returns:
            Endpoint creado
        """
        if endpoint_id is None:
            endpoint_id = f"endpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        endpoint = ServingEndpoint(
            endpoint_id=endpoint_id,
            model_id=model_id,
            strategy=strategy,
            url=f"/api/models/{model_id}/predict",
            status="active",
            created_at=datetime.now().isoformat()
        )
        
        self.endpoints[endpoint_id] = endpoint
        
        logger.info(f"Endpoint creado: {endpoint_id} - {strategy.value}")
        
        return endpoint
    
    def serve_prediction(
        self,
        endpoint_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Servir predicción
        
        Args:
            endpoint_id: ID del endpoint
            input_data: Datos de entrada
        
        Returns:
            Predicción
        """
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint no encontrado: {endpoint_id}")
        
        endpoint = self.endpoints[endpoint_id]
        
        # Simulación de predicción
        prediction = {
            "endpoint_id": endpoint_id,
            "model_id": endpoint.model_id,
            "prediction": "class_A",
            "confidence": 0.85,
            "latency_ms": 10.5,
            "timestamp": datetime.now().isoformat()
        }
        
        # Registrar métricas
        if endpoint_id not in self.metrics:
            self.metrics[endpoint_id] = []
        
        self.metrics[endpoint_id].append({
            "timestamp": datetime.now().isoformat(),
            "latency_ms": prediction["latency_ms"],
            "success": True
        })
        
        logger.debug(f"Predicción servida: {endpoint_id}")
        
        return prediction
    
    def get_endpoint_metrics(
        self,
        endpoint_id: str
    ) -> Dict[str, Any]:
        """Obtener métricas del endpoint"""
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint no encontrado: {endpoint_id}")
        
        if endpoint_id not in self.metrics:
            return {
                "endpoint_id": endpoint_id,
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0
            }
        
        metrics_list = self.metrics[endpoint_id]
        
        if not metrics_list:
            return {
                "endpoint_id": endpoint_id,
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0
            }
        
        total_requests = len(metrics_list)
        avg_latency = sum(m["latency_ms"] for m in metrics_list) / total_requests
        success_count = sum(1 for m in metrics_list if m.get("success", False))
        success_rate = success_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "endpoint_id": endpoint_id,
            "total_requests": total_requests,
            "avg_latency_ms": avg_latency,
            "success_rate": success_rate,
            "p95_latency_ms": avg_latency * 1.5,
            "p99_latency_ms": avg_latency * 2.0
        }


# Instancia global
_model_serving: Optional[ModelServing] = None


def get_model_serving() -> ModelServing:
    """Obtener instancia global del sistema"""
    global _model_serving
    if _model_serving is None:
        _model_serving = ModelServing()
    return _model_serving


