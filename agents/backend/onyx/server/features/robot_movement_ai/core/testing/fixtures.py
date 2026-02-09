"""
Test Fixtures
=============

Fixtures para testing.
"""

from typing import Dict, Any, List, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_mock_route(
    start: str = "A",
    end: str = "B",
    path: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Crear ruta mock.
    
    Args:
        start: Nodo de inicio
        end: Nodo de fin
        path: Ruta (opcional)
        metrics: Métricas (opcional)
        
    Returns:
        Diccionario con ruta mock
    """
    return {
        "start_node": start,
        "end_node": end,
        "route": path or [start, "intermediate", end],
        "metrics": metrics or {
            "distance": 10.0,
            "time": 5.0,
            "cost": 2.0,
            "efficiency": 0.8,
            "reliability": 0.9
        },
        "confidence": 0.9,
        "metadata": {}
    }


def create_mock_graph(
    num_nodes: int = 5,
    num_edges: int = 10
) -> Dict[str, Any]:
    """
    Crear grafo mock.
    
    Args:
        num_nodes: Número de nodos
        num_edges: Número de aristas
        
    Returns:
        Diccionario con grafo mock
    """
    nodes = {f"node_{i}": {"id": f"node_{i}", "position": (i, i)} for i in range(num_nodes)}
    
    edges = []
    for i in range(num_edges):
        source = f"node_{i % num_nodes}"
        target = f"node_{(i + 1) % num_nodes}"
        edges.append({
            "source": source,
            "target": target,
            "weight": float(i + 1)
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {}
    }


def create_mock_model(
    input_dim: int = 20,
    output_dim: int = 4
) -> Any:
    """
    Crear modelo mock.
    
    Args:
        input_dim: Dimensión de entrada
        output_dim: Dimensión de salida
        
    Returns:
        Modelo mock
    """
    if not TORCH_AVAILABLE:
        # Mock simple sin PyTorch
        class MockModel:
            def forward(self, x):
                return np.random.randn(output_dim)
            def predict(self, features):
                return {
                    "predicted_time": 5.0,
                    "predicted_cost": 2.0,
                    "predicted_load": 0.5,
                    "success_probability": 0.9
                }
        return MockModel()
    
    # Modelo PyTorch
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.linear(x)
        
        def predict(self, features):
            with torch.no_grad():
                x = torch.FloatTensor([features.get(f"feature_{i}", 0.0) for i in range(input_dim)])
                output = self.forward(x)
                return {
                    "predicted_time": float(output[0]),
                    "predicted_cost": float(output[1]),
                    "predicted_load": float(output[2]),
                    "success_probability": float(torch.sigmoid(output[3]))
                }
    
    return MockModel()


def create_mock_dataset(
    num_samples: int = 100,
    input_dim: int = 20,
    output_dim: int = 4
) -> Dict[str, Any]:
    """
    Crear dataset mock.
    
    Args:
        num_samples: Número de muestras
        input_dim: Dimensión de entrada
        output_dim: Dimensión de salida
        
    Returns:
        Diccionario con dataset mock
    """
    if TORCH_AVAILABLE:
        import torch
        return {
            "features": torch.randn(num_samples, input_dim),
            "targets": torch.randn(num_samples, output_dim)
        }
    else:
        return {
            "features": np.random.randn(num_samples, input_dim),
            "targets": np.random.randn(num_samples, output_dim)
        }

