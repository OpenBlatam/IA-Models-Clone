"""
Test Assertions
===============

Assertions personalizadas para testing.
"""

from typing import Dict, Any, List


def assert_route_valid(route: Dict[str, Any]):
    """
    Assert que una ruta es válida.
    
    Args:
        route: Diccionario con ruta
        
    Raises:
        AssertionError: Si la ruta no es válida
    """
    assert "route" in route, "Ruta debe tener campo 'route'"
    assert isinstance(route["route"], list), "Campo 'route' debe ser lista"
    assert len(route["route"]) >= 2, "Ruta debe tener al menos 2 nodos"
    
    assert "metrics" in route, "Ruta debe tener campo 'metrics'"
    assert isinstance(route["metrics"], dict), "Campo 'metrics' debe ser diccionario"
    
    required_metrics = ["distance", "time", "cost"]
    for metric in required_metrics:
        assert metric in route["metrics"], f"Métrica requerida '{metric}' no encontrada"
        assert isinstance(route["metrics"][metric], (int, float)), f"Métrica '{metric}' debe ser numérica"
    
    assert "confidence" in route, "Ruta debe tener campo 'confidence'"
    assert 0.0 <= route["confidence"] <= 1.0, "Confidence debe estar entre 0 y 1"


def assert_metrics_valid(metrics: Dict[str, float]):
    """
    Assert que métricas son válidas.
    
    Args:
        metrics: Diccionario con métricas
        
    Raises:
        AssertionError: Si las métricas no son válidas
    """
    assert isinstance(metrics, dict), "Métricas deben ser diccionario"
    
    required = ["distance", "time", "cost"]
    for metric in required:
        assert metric in metrics, f"Métrica requerida '{metric}' no encontrada"
        assert isinstance(metrics[metric], (int, float)), f"Métrica '{metric}' debe ser numérica"
        assert metrics[metric] >= 0, f"Métrica '{metric}' debe ser no negativa"


def assert_model_output_valid(output: Dict[str, float]):
    """
    Assert que output de modelo es válido.
    
    Args:
        output: Diccionario con output
        
    Raises:
        AssertionError: Si el output no es válido
    """
    assert isinstance(output, dict), "Output debe ser diccionario"
    
    # Verificar que todos los valores son numéricos
    for key, value in output.items():
        assert isinstance(value, (int, float)), f"Valor '{key}' debe ser numérico"
    
    # Verificar rangos comunes
    if "confidence" in output:
        assert 0.0 <= output["confidence"] <= 1.0, "Confidence debe estar entre 0 y 1"
    
    if "probability" in output:
        assert 0.0 <= output["probability"] <= 1.0, "Probability debe estar entre 0 y 1"

