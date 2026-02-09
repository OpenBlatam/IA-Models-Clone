"""
Layer Architecture Metrics

Functions for calculating complexity, impact, cohesion, and other metrics
for layer architecture analysis.
"""

from functools import lru_cache
from typing import Dict, Any, List, Tuple, Callable, Optional

from ..constants import LayerName, VALID_LAYERS
from ..utils import get_layer_order, get_layer_count
from ..analysis import (
    get_layer_dependencies,
    get_dependent_layers,
    get_layer_depth,
    get_layer_descendants,
)


def get_layer_complexity_score(layer_name: LayerName) -> float:
    """
    Calculate a complexity score for a layer based on its dependencies and dependents.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        float: Complexity score (higher = more complex)
    """
    dep_count = len(get_layer_dependencies(layer_name))
    dependent_count = len(get_dependent_layers(layer_name))
    depth = get_layer_depth(layer_name)
    
    return dep_count + dependent_count + depth


def get_layer_impact_score(layer_name: LayerName) -> float:
    """
    Calculate impact score: how many layers would be affected if this layer changes.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        float: Impact score (number of dependent layers)
    """
    return float(len(get_layer_descendants(layer_name)))


def get_layer_cohesion_score(layer_name: LayerName) -> float:
    """
    Calculate cohesion score for a layer (how well it's organized).
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        float: Cohesion score (0.0 to 1.0, higher = more cohesive)
    """
    deps_count = len(get_layer_dependencies(layer_name))
    dependents_count = len(get_dependent_layers(layer_name))
    
    # Lower dependencies and higher dependents = better cohesion
    if deps_count == 0:
        return 1.0
    
    cohesion = 1.0 / (deps_count + 1)
    return min(cohesion, 1.0)


def calculate_layer_coupling(layer1: LayerName, layer2: LayerName) -> float:
    """
    Calculate coupling score between two layers (0.0 to 1.0).
    
    Args:
        layer1: First layer
        layer2: Second layer
        
    Returns:
        float: Coupling score (higher = more coupled)
    """
    from ..utils import can_access_layer
    from ..analysis import get_layer_distance
    
    if layer1 == layer2:
        return 1.0
    
    if can_access_layer(layer1, layer2):
        distance = get_layer_distance(layer1, layer2)
        if distance is not None:
            return 1.0 / (distance + 1)
    
    return 0.0


def count_layer_connections(layer_name: LayerName) -> Dict[str, int]:
    """
    Count all connections (dependencies and dependents) for a layer.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Dict[str, int]: Connection counts
    """
    deps = len(get_layer_dependencies(layer_name))
    dependents = len(get_dependent_layers(layer_name))
    return {
        "dependencies": deps,
        "dependents": dependents,
        "total": deps + dependents,
    }


def get_all_layer_metrics() -> Dict[LayerName, Dict[str, Any]]:
    """
    Get comprehensive metrics for all layers at once.
    
    Returns:
        Dict[LayerName, Dict[str, Any]]: Dictionary mapping each layer to its metrics
    """
    return {
        layer: {
            "order": get_layer_order(layer),
            "depth": get_layer_depth(layer),
            "complexity": get_layer_complexity_score(layer),
            "impact": get_layer_impact_score(layer),
            "cohesion": get_layer_cohesion_score(layer),
            "connections": count_layer_connections(layer),
        }
        for layer in VALID_LAYERS
    }


def get_most_complex_layers(limit: int = 3) -> List[Tuple[LayerName, float]]:
    """
    Get the most complex layers sorted by complexity score.
    
    Args:
        limit: Maximum number of layers to return
        
    Returns:
        List[Tuple[LayerName, float]]: List of (layer, score) tuples
    """
    scores = [
        (layer, get_layer_complexity_score(layer))
        for layer in VALID_LAYERS
    ]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:limit]


def get_least_complex_layers(limit: int = 3) -> List[Tuple[LayerName, float]]:
    """
    Get the least complex layers sorted by complexity score.
    
    Args:
        limit: Maximum number of layers to return
        
    Returns:
        List[Tuple[LayerName, float]]: List of (layer, score) tuples
    """
    scores = [
        (layer, get_layer_complexity_score(layer))
        for layer in VALID_LAYERS
    ]
    return sorted(scores, key=lambda x: x[1])[:limit]


def get_high_impact_layers(threshold: float = 2.0) -> Tuple[LayerName, ...]:
    """
    Get layers with high impact (many dependents).
    
    Args:
        threshold: Minimum impact score
        
    Returns:
        Tuple[LayerName, ...]: High impact layers
    """
    return tuple(
        sorted(
            (
                layer for layer in VALID_LAYERS
                if get_layer_impact_score(layer) >= threshold
            ),
            key=get_layer_impact_score,
            reverse=True
        )
    )


def get_layer_statistics() -> Dict[str, Any]:
    """
    Get statistics about the layer architecture.
    
    Returns:
        Dict[str, Any]: Statistics including counts, depths, etc.
    """
    from ..analysis import get_all_layer_dependencies, get_transitive_dependencies
    
    all_deps = get_all_layer_dependencies()
    max_depth = max(
        len(get_transitive_dependencies(layer))
        for layer in VALID_LAYERS
    )
    
    total_dependencies = sum(
        len(deps) for deps in all_deps.values()
    )
    
    return {
        "total_layers": get_layer_count(),
        "max_dependency_depth": max_depth,
        "total_dependencies": total_dependencies,
        "average_dependencies": total_dependencies / get_layer_count(),
        "layers_with_no_dependencies": len([
            layer for layer, deps in all_deps.items()
            if len(deps) == 0
        ]),
        "has_cycles": False,  # Clean architecture should not have cycles
    }


__all__ = [
    "get_layer_complexity_score",
    "get_layer_impact_score",
    "get_layer_cohesion_score",
    "calculate_layer_coupling",
    "count_layer_connections",
    "get_all_layer_metrics",
    "get_most_complex_layers",
    "get_least_complex_layers",
    "get_high_impact_layers",
    "get_layer_statistics",
]



