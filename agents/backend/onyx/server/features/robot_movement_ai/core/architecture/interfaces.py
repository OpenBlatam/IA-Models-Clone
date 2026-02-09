"""
Interfaces and Protocols
========================

Definición de interfaces y protocolos para el sistema de routing.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Dict, Any, List, Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class RouteRequest:
    """Request para encontrar una ruta."""
    start_node: str
    end_node: str
    strategy: str = "shortest_path"
    constraints: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


@dataclass
class RouteResponse:
    """Response con la ruta encontrada."""
    route: List[str]
    metrics: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any] = None


@runtime_checkable
class IRouteModel(Protocol):
    """Interface para modelos de routing."""
    
    def forward(self, x: Any) -> Any:
        """Forward pass."""
        ...
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predecir métricas de ruta."""
        ...
    
    def train(self, data: Any) -> Dict[str, Any]:
        """Entrenar modelo."""
        ...
    
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Evaluar modelo."""
        ...


@runtime_checkable
class IRouteStrategy(Protocol):
    """Interface para estrategias de routing."""
    
    def find_route(
        self,
        start: str,
        end: str,
        graph: Any,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RouteResponse:
        """Encontrar ruta."""
        ...
    
    def get_name(self) -> str:
        """Obtener nombre de la estrategia."""
        ...


@runtime_checkable
class IRouteRepository(Protocol):
    """Interface para repositorio de rutas."""
    
    def save_route(self, route: RouteResponse) -> str:
        """Guardar ruta."""
        ...
    
    def get_route(self, route_id: str) -> Optional[RouteResponse]:
        """Obtener ruta por ID."""
        ...
    
    def find_routes(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100
    ) -> List[RouteResponse]:
        """Buscar rutas."""
        ...


@runtime_checkable
class IRouteService(Protocol):
    """Interface para servicio de routing."""
    
    def find_route(self, request: RouteRequest) -> RouteResponse:
        """Encontrar ruta."""
        ...
    
    def find_multiple_routes(
        self,
        requests: List[RouteRequest]
    ) -> List[RouteResponse]:
        """Encontrar múltiples rutas."""
        ...


@runtime_checkable
class IDataProcessor(Protocol):
    """Interface para procesamiento de datos."""
    
    def preprocess(self, data: Any) -> Any:
        """Preprocesar datos."""
        ...
    
    def postprocess(self, data: Any) -> Any:
        """Postprocesar datos."""
        ...
    
    def validate(self, data: Any) -> bool:
        """Validar datos."""
        ...


@runtime_checkable
class ITrainingPipeline(Protocol):
    """Interface para pipeline de entrenamiento."""
    
    def train(
        self,
        model: Any,
        train_data: Any,
        val_data: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Entrenar modelo."""
        ...
    
    def evaluate(
        self,
        model: Any,
        test_data: Any
    ) -> Dict[str, float]:
        """Evaluar modelo."""
        ...


@runtime_checkable
class IInferenceEngine(Protocol):
    """Interface para motor de inferencia."""
    
    def predict(
        self,
        model: Any,
        input_data: Any,
        batch_size: Optional[int] = None
    ) -> Any:
        """Predecir."""
        ...
    
    def predict_batch(
        self,
        model: Any,
        batch: Any
    ) -> Any:
        """Predecir batch."""
        ...


# Abstract Base Classes para implementaciones concretas

class BaseRouteModel(ABC):
    """Clase base abstracta para modelos de routing."""
    
    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass."""
        pass
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predecir métricas."""
        pass
    
    def train(self, data: Any) -> Dict[str, Any]:
        """Entrenar modelo (opcional)."""
        raise NotImplementedError
    
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Evaluar modelo (opcional)."""
        raise NotImplementedError


class BaseRouteStrategy(ABC):
    """Clase base abstracta para estrategias de routing."""
    
    @abstractmethod
    def find_route(
        self,
        start: str,
        end: str,
        graph: Any,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RouteResponse:
        """Encontrar ruta."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Obtener nombre."""
        pass

