"""
Layer Architecture Validators

Functions for validating layer access, dependencies, and architecture integrity.
"""

from functools import lru_cache
from typing import Optional, Tuple, List, Dict

from ..constants import LayerName, VALID_LAYERS, LAYER_DEPENDENCY_RULES
from ..utils import (
    can_access_layer,
    get_layer_dependencies,
    is_valid_layer,
)
from ..exceptions import LayerAccessError, InvalidLayerError


def validate_layer_access(source_layer: LayerName, target_layer: LayerName) -> Tuple[bool, Optional[str]]:
    """
    Validate layer access and return detailed result with reason.
    
    Args:
        source_layer: The layer trying to access
        target_layer: The layer being accessed
        
    Returns:
        Tuple[bool, Optional[str]]: (is_allowed, reason)
    """
    if source_layer == target_layer:
        return True, None
    
    if can_access_layer(source_layer, target_layer):
        return True, None
    
    return False, (
        f"{source_layer.capitalize()} layer cannot access "
        f"{target_layer.capitalize()} layer (violates dependency rule)"
    )


def require_layer_access(source_layer: LayerName, target_layer: LayerName) -> None:
    """
    Require that source_layer can access target_layer, raising exception if not.
    
    Args:
        source_layer: The layer trying to access
        target_layer: The layer being accessed
        
    Raises:
        LayerAccessError: If access is not allowed
    """
    is_allowed, reason = validate_layer_access(source_layer, target_layer)
    if not is_allowed:
        raise LayerAccessError(source_layer, target_layer)


def require_valid_layer(layer_name: str) -> LayerName:
    """
    Require that layer_name is valid, raising exception if not.
    
    Args:
        layer_name: Name of the layer to validate
        
    Returns:
        LayerName: Validated layer name
        
    Raises:
        InvalidLayerError: If layer name is invalid
    """
    if not is_valid_layer(layer_name):
        raise InvalidLayerError(layer_name)
    return layer_name


def get_layer_dependency_graph() -> Dict[LayerName, List[LayerName]]:
    """
    Get a dependency graph showing which layers each layer depends on.
    
    Returns:
        Dict[LayerName, List[LayerName]]: Dependency graph as adjacency list
    """
    return {
        layer: list(get_layer_dependencies(layer))
        for layer in VALID_LAYERS
    }


def check_dependency_cycle() -> Optional[List[LayerName]]:
    """
    Check if there are any circular dependencies between layers.
    
    Returns:
        Optional[List[LayerName]]: Cycle found, or None if no cycles
    """
    graph = get_layer_dependency_graph()
    visited = set()
    rec_stack = set()
    
    def has_cycle(node: LayerName) -> Optional[List[LayerName]]:
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                cycle = has_cycle(neighbor)
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                return [node, neighbor]
        
        rec_stack.remove(node)
        return None
    
    for layer in VALID_LAYERS:
        if layer not in visited:
            cycle = has_cycle(layer)
            if cycle:
                return cycle
    
    return None


def validate_layer_chain(layers: Tuple[LayerName, ...]) -> Tuple[bool, Optional[str]]:
    """
    Validate that a chain of layers follows proper dependency order.
    
    Args:
        layers: Tuple of layers to validate
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if len(layers) < 2:
        return True, None
    
    for i in range(len(layers) - 1):
        source = layers[i]
        target = layers[i + 1]
        
        if not can_access_layer(source, target):
            return False, (
                f"Invalid chain: {source} cannot access {target} "
                f"(position {i} -> {i+1})"
            )
    
    return True, None


def validate_architecture() -> Tuple[bool, List[str]]:
    """
    Perform comprehensive validation of the entire architecture.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    issues = []
    
    cycle = check_dependency_cycle()
    if cycle:
        issues.append(f"Circular dependency detected: {' -> '.join(cycle)}")
    
    for layer in VALID_LAYERS:
        deps = get_layer_dependencies(layer)
        for dep in deps:
            if not is_valid_layer(dep):
                issues.append(f"{layer} depends on invalid layer: {dep}")
    
    all_layers_valid = all(is_valid_layer(layer) for layer in VALID_LAYERS)
    if not all_layers_valid:
        issues.append("Some layers are invalid")
    
    return len(issues) == 0, issues


__all__ = [
    "validate_layer_access",
    "require_layer_access",
    "require_valid_layer",
    "get_layer_dependency_graph",
    "check_dependency_cycle",
    "validate_layer_chain",
    "validate_architecture",
]

