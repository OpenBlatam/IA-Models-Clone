"""
Layer Architecture Analysis

Advanced analysis functions for layer relationships, dependencies, and structure.
"""

from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any, Callable

from ..constants import LayerName, VALID_LAYERS
from ..utils import (
    get_layer_dependencies,
    get_layer_order,
    can_access_layer,
    get_layers_below,
    get_layers_above,
    get_layer_hierarchy,
)
from ..validators import get_layer_dependency_graph


def find_accessible_layers(source_layer: LayerName) -> Tuple[LayerName, ...]:
    """
    Find all layers that the source layer can access.
    
    Args:
        source_layer: The layer to check
        
    Returns:
        Tuple[LayerName, ...]: All layers that can be accessed
    """
    dependencies = get_layer_dependencies(source_layer)
    return (source_layer,) + dependencies


def find_layers_that_can_access(target_layer: LayerName) -> Tuple[LayerName, ...]:
    """
    Find all layers that can access the target layer.
    
    Args:
        target_layer: The layer being accessed
        
    Returns:
        Tuple[LayerName, ...]: All layers that can access this layer
    """
    return tuple(
        layer for layer in VALID_LAYERS
        if can_access_layer(layer, target_layer)
    )


def get_transitive_dependencies(layer_name: LayerName) -> Tuple[LayerName, ...]:
    """
    Get all transitive dependencies of a layer (dependencies of dependencies).
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple[LayerName, ...]: All transitive dependencies
    """
    direct_deps = get_layer_dependencies(layer_name)
    transitive = set(direct_deps)
    
    for dep in direct_deps:
        transitive.update(get_transitive_dependencies(dep))
    
    return tuple(sorted(transitive, key=get_layer_order))


def get_layer_path(source_layer: LayerName, target_layer: LayerName) -> Optional[List[LayerName]]:
    """
    Get the dependency path from source_layer to target_layer if accessible.
    
    Args:
        source_layer: Starting layer
        target_layer: Target layer
        
    Returns:
        Optional[List[LayerName]]: Path if accessible, None otherwise
    """
    if not can_access_layer(source_layer, target_layer):
        return None
    
    if source_layer == target_layer:
        return [source_layer]
    
    path = [source_layer]
    current = source_layer
    
    while current != target_layer:
        deps = get_layer_dependencies(current)
        if target_layer in deps:
            path.append(target_layer)
            break
        elif deps:
            current = deps[0]
            path.append(current)
        else:
            return None
    
    return path


def get_layer_distance(layer1: LayerName, layer2: LayerName) -> Optional[int]:
    """
    Get the distance (number of steps) between two layers in the dependency graph.
    
    Args:
        layer1: First layer
        layer2: Second layer
        
    Returns:
        Optional[int]: Distance if reachable, None otherwise
    """
    path = get_layer_path(layer1, layer2)
    if path is None:
        return None
    return len(path) - 1


def get_layer_ancestors(layer_name: LayerName) -> Tuple[LayerName, ...]:
    """
    Get all ancestor layers (layers that this layer depends on, recursively).
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple[LayerName, ...]: All ancestor layers
    """
    return get_transitive_dependencies(layer_name)


def get_dependent_layers(layer_name: LayerName) -> Tuple[LayerName, ...]:
    """
    Get all layers that depend on the given layer.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple[LayerName, ...]: Layers that depend on this layer
    """
    return tuple(
        layer for layer in VALID_LAYERS
        if layer_name in get_layer_dependencies(layer)
    )


def get_layer_descendants(layer_name: LayerName) -> Tuple[LayerName, ...]:
    """
    Get all descendant layers (layers that depend on this layer, recursively).
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple[LayerName, ...]: All descendant layers
    """
    descendants = set()
    
    def collect_descendants(layer: LayerName):
        for dependent in get_dependent_layers(layer):
            descendants.add(dependent)
            collect_descendants(dependent)
    
    collect_descendants(layer_name)
    return tuple(sorted(descendants, key=get_layer_order))


def get_layer_relationships(layer_name: LayerName) -> Dict[str, Any]:
    """
    Get all relationships for a layer (dependencies, dependents, etc.).
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Dict[str, Any]: Dictionary with all relationships
    """
    return {
        "dependencies": get_layer_dependencies(layer_name),
        "dependents": get_dependent_layers(layer_name),
        "ancestors": get_layer_ancestors(layer_name),
        "descendants": get_layer_descendants(layer_name),
        "accessible_layers": find_accessible_layers(layer_name),
        "layers_that_can_access": find_layers_that_can_access(layer_name),
        "layers_below": get_layers_below(layer_name),
        "layers_above": get_layers_above(layer_name),
    }


def get_layer_depth(layer_name: LayerName) -> int:
    """
    Get the maximum depth of dependencies for a layer.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        int: Maximum dependency depth (0 = no dependencies)
    """
    deps = get_layer_dependencies(layer_name)
    if not deps:
        return 0
    
    return 1 + max(
        get_layer_depth(dep) for dep in deps
    )


def get_nearest_common_ancestor(
    layer1: LayerName,
    layer2: LayerName
) -> Optional[LayerName]:
    """
    Get the nearest common ancestor of two layers.
    
    Args:
        layer1: First layer
        layer2: Second layer
        
    Returns:
        Optional[LayerName]: Nearest common ancestor, or None
    """
    ancestors1 = set(get_layer_ancestors(layer1))
    ancestors2 = set(get_layer_ancestors(layer2))
    common = ancestors1 & ancestors2
    
    if not common:
        return None
    
    return max(common, key=get_layer_order)


def layer_to_dict(layer_name: LayerName) -> Dict[str, Any]:
    """
    Convert a layer to a dictionary representation with all its information.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Dict[str, Any]: Dictionary with layer information
    """
    from ..utils import get_layer_description
    
    return {
        "name": layer_name,
        "order": get_layer_order(layer_name),
        "description": get_layer_description(layer_name),
        "dependencies": get_layer_dependencies(layer_name),
        "accessible_layers": find_accessible_layers(layer_name),
        "layers_below": get_layers_below(layer_name),
        "layers_above": get_layers_above(layer_name),
        "transitive_dependencies": get_transitive_dependencies(layer_name),
    }


def get_all_layers_info() -> Dict[LayerName, Dict[str, Any]]:
    """
    Get information about all layers in a structured format.
    
    Returns:
        Dict[LayerName, Dict[str, Any]]: Dictionary mapping each layer to its info
    """
    return {
        layer: layer_to_dict(layer)
        for layer in VALID_LAYERS
    }


def get_all_layer_dependencies() -> Dict[LayerName, Tuple[LayerName, ...]]:
    """
    Get dependencies for all layers at once.
    
    Returns:
        Dict[LayerName, Tuple[LayerName, ...]]: Dictionary mapping each layer to its dependencies
    """
    return {
        layer: get_layer_dependencies(layer)
        for layer in VALID_LAYERS
    }


__all__ = [
    "find_accessible_layers",
    "find_layers_that_can_access",
    "get_transitive_dependencies",
    "get_layer_path",
    "get_layer_distance",
    "get_layer_ancestors",
    "get_dependent_layers",
    "get_layer_descendants",
    "get_layer_relationships",
    "get_layer_depth",
    "get_nearest_common_ancestor",
    "layer_to_dict",
    "get_all_layers_info",
    "get_all_layer_dependencies",
]



