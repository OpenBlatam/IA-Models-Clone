"""
Pydantic Schemas
================

Esquemas de validación usando Pydantic.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class RouteRequestSchema(BaseModel):
    """Schema para request de ruta."""
    start_node: str = Field(..., min_length=1, description="Nodo de inicio")
    end_node: str = Field(..., min_length=1, description="Nodo de fin")
    strategy: str = Field(default="shortest_path", description="Estrategia de routing")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Restricciones")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadatos")
    
    @validator('start_node', 'end_node')
    def validate_nodes(cls, v):
        """Validar que los nodos no estén vacíos."""
        if not v or not v.strip():
            raise ValueError("Los nodos no pueden estar vacíos")
        return v.strip()
    
    @validator('strategy')
    def validate_strategy(cls, v):
        """Validar estrategia."""
        valid_strategies = [
            "shortest_path", "fastest_path", "least_cost",
            "load_balanced", "adaptive", "deep_learning",
            "transformer", "llm_optimized", "gnn_based",
            "reinforcement_learning"
        ]
        if v not in valid_strategies:
            raise ValueError(f"Estrategia '{v}' no válida. Válidas: {valid_strategies}")
        return v
    
    class Config:
        """Configuración de Pydantic."""
        json_schema_extra = {
            "example": {
                "start_node": "A",
                "end_node": "B",
                "strategy": "shortest_path",
                "constraints": {"max_distance": 100}
            }
        }


class RouteResponseSchema(BaseModel):
    """Schema para response de ruta."""
    route: List[str] = Field(..., min_items=2, description="Ruta encontrada")
    metrics: Dict[str, float] = Field(..., description="Métricas de la ruta")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la ruta")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadatos")
    
    @validator('route')
    def validate_route(cls, v):
        """Validar que la ruta tenga al menos 2 nodos."""
        if len(v) < 2:
            raise ValueError("La ruta debe tener al menos 2 nodos")
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validar métricas."""
        required_metrics = ['distance', 'time', 'cost']
        for metric in required_metrics:
            if metric not in v:
                raise ValueError(f"Métrica requerida '{metric}' no encontrada")
        return v
    
    class Config:
        """Configuración de Pydantic."""
        json_schema_extra = {
            "example": {
                "route": ["A", "B", "C"],
                "metrics": {
                    "distance": 10.0,
                    "time": 5.0,
                    "cost": 2.0
                },
                "confidence": 0.9
            }
        }


class ModelConfigSchema(BaseModel):
    """Schema para configuración de modelo."""
    input_dim: int = Field(..., gt=0, description="Dimensión de entrada")
    hidden_dims: List[int] = Field(..., min_items=1, description="Dimensiones ocultas")
    output_dim: int = Field(..., gt=0, description="Dimensión de salida")
    dropout: float = Field(default=0.2, ge=0.0, le=1.0, description="Tasa de dropout")
    activation: str = Field(default="relu", description="Función de activación")
    use_batch_norm: bool = Field(default=True, description="Usar batch normalization")
    
    @validator('hidden_dims')
    def validate_hidden_dims(cls, v):
        """Validar dimensiones ocultas."""
        for dim in v:
            if dim <= 0:
                raise ValueError("Las dimensiones ocultas deben ser positivas")
        return v
    
    @validator('activation')
    def validate_activation(cls, v):
        """Validar función de activación."""
        valid_activations = ['relu', 'tanh', 'sigmoid', 'gelu', 'swish']
        if v not in valid_activations:
            raise ValueError(f"Activación '{v}' no válida. Válidas: {valid_activations}")
        return v


class TrainingConfigSchema(BaseModel):
    """Schema para configuración de entrenamiento."""
    epochs: int = Field(..., gt=0, description="Número de épocas")
    batch_size: int = Field(..., gt=0, description="Tamaño de batch")
    learning_rate: float = Field(..., gt=0.0, description="Learning rate")
    weight_decay: float = Field(default=1e-5, ge=0.0, description="Weight decay")
    optimizer: str = Field(default="adam", description="Optimizador")
    scheduler: Optional[str] = Field(default=None, description="Scheduler")
    early_stopping_patience: int = Field(default=10, ge=0, description="Patience para early stopping")
    use_mixed_precision: bool = Field(default=True, description="Usar mixed precision")
    
    @validator('optimizer')
    def validate_optimizer(cls, v):
        """Validar optimizador."""
        valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        if v not in valid_optimizers:
            raise ValueError(f"Optimizador '{v}' no válido. Válidos: {valid_optimizers}")
        return v


class NodeSchema(BaseModel):
    """Schema para nodo."""
    id: str = Field(..., min_length=1, description="ID del nodo")
    position: Optional[tuple] = Field(default=None, description="Posición")
    features: Dict[str, Any] = Field(default_factory=dict, description="Features")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos")


class EdgeSchema(BaseModel):
    """Schema para arista."""
    source: str = Field(..., min_length=1, description="Nodo origen")
    target: str = Field(..., min_length=1, description="Nodo destino")
    weight: float = Field(default=1.0, ge=0.0, description="Peso")
    features: Dict[str, Any] = Field(default_factory=dict, description="Features")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos")


class GraphSchema(BaseModel):
    """Schema para grafo."""
    nodes: Dict[str, NodeSchema] = Field(default_factory=dict, description="Nodos")
    edges: List[EdgeSchema] = Field(default_factory=list, description="Aristas")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos")
    
    @validator('edges')
    def validate_edges(cls, v, values):
        """Validar que las aristas referencien nodos existentes."""
        nodes = values.get('nodes', {})
        for edge in v:
            if edge.source not in nodes:
                raise ValueError(f"Nodo origen '{edge.source}' no existe")
            if edge.target not in nodes:
                raise ValueError(f"Nodo destino '{edge.target}' no existe")
        return v

