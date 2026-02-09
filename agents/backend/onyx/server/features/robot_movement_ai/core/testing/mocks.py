"""
Mock Objects
============

Objetos mock para testing.
"""

from typing import Dict, Any, List, Optional
from core.architecture.interfaces import IRouteStrategy, IRouteModel, IInferenceEngine
from core.architecture.interfaces import RouteRequest, RouteResponse


class MockRouteStrategy(IRouteStrategy):
    """Mock de estrategia de routing."""
    
    def __init__(self, name: str = "mock"):
        """Inicializar mock."""
        self.name = name
    
    def find_route(
        self,
        start: str,
        end: str,
        graph: Any,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RouteResponse:
        """Encontrar ruta mock."""
        return RouteResponse(
            route=[start, "intermediate", end],
            metrics={
                "distance": 10.0,
                "time": 5.0,
                "cost": 2.0
            },
            confidence=0.9
        )
    
    def get_name(self) -> str:
        """Obtener nombre."""
        return self.name


class MockRouteModel(IRouteModel):
    """Mock de modelo de routing."""
    
    def forward(self, x: Any) -> Any:
        """Forward mock."""
        import numpy as np
        return np.random.randn(4)
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predict mock."""
        return {
            "predicted_time": 5.0,
            "predicted_cost": 2.0,
            "predicted_load": 0.5,
            "success_probability": 0.9
        }
    
    def train(self, data: Any) -> Dict[str, Any]:
        """Train mock."""
        return {"loss": 0.1, "epoch": 1}
    
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Evaluate mock."""
        return {"accuracy": 0.9, "loss": 0.1}


class MockInferenceEngine(IInferenceEngine):
    """Mock de motor de inferencia."""
    
    def predict(
        self,
        model: Any,
        input_data: Any,
        batch_size: Optional[int] = None
    ) -> Any:
        """Predict mock."""
        import numpy as np
        return np.random.randn(4)
    
    def predict_batch(
        self,
        model: Any,
        batch: Any
    ) -> Any:
        """Predict batch mock."""
        import numpy as np
        batch_size = len(batch) if hasattr(batch, '__len__') else 1
        return np.random.randn(batch_size, 4)

