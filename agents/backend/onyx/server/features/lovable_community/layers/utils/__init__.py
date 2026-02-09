"""
Layer Architecture Utilities

Basic utility functions for layer information, validation, and basic operations.
"""

from functools import lru_cache
from typing import Optional, Tuple, TypeGuard, Dict, List

from ..constants import (
    LayerName,
    LayerInfo,
    ModuleInfo,
    VALID_LAYERS,
    LAYER_DESCRIPTIONS,
    LAYER_DEPENDENCY_RULES,
    LAYER_ORDER,
    LayerOrder,
)

__version__ = "1.0.0"
__author__ = "Lovable Community Team"
__description__ = "Clean Architecture implementation with layered separation"


@lru_cache(maxsize=1)
def get_layer_info() -> LayerInfo:
    """
    Get information about all layers in the architecture.
    
    Returns:
        LayerInfo: Dictionary mapping layer names to their descriptions
    """
    return LAYER_DESCRIPTIONS.copy()


@lru_cache(maxsize=1)
def get_module_info() -> ModuleInfo:
    """
    Get module metadata information.
    
    Returns:
        ModuleInfo: Dictionary containing module version, author, and description
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
    }


def is_valid_layer(layer_name: str) -> TypeGuard[LayerName]:
    """
    Check if a layer name is valid using type guard for better type inference.
    
    Args:
        layer_name: Name of the layer to validate
        
    Returns:
        TypeGuard[LayerName]: Type guard that narrows the type if True
    """
    return isinstance(layer_name, str) and layer_name in VALID_LAYERS


@lru_cache(maxsize=1)
def get_all_layers() -> Tuple[str, ...]:
    """
    Get a tuple of all valid layer names.
    
    Returns:
        Tuple[str, ...]: Tuple of all valid layer names, sorted alphabetically
    """
    return tuple(sorted(VALID_LAYERS))


@lru_cache(maxsize=4)
def get_layer_description(layer_name: LayerName) -> Optional[str]:
    """
    Get the description for a specific layer.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Optional[str]: Layer description if valid, None otherwise
    """
    if not isinstance(layer_name, str):
        raise ValueError(f"layer_name must be a string, got {type(layer_name).__name__}")
    
    if not is_valid_layer(layer_name):
        return None
    
    layer_info = get_layer_info()
    return layer_info.get(layer_name)


@lru_cache(maxsize=1)
def get_layer_count() -> int:
    """
    Get the total number of layers in the architecture.
    
    Returns:
        int: Total number of layers
    """
    return len(VALID_LAYERS)


def validate_layers(layer_names: Tuple[str, ...]) -> Dict[str, bool]:
    """
    Validate multiple layer names at once.
    
    Args:
        layer_names: Tuple of layer names to validate
        
    Returns:
        Dict[str, bool]: Dictionary mapping layer names to their validation status
    """
    if not isinstance(layer_names, tuple):
        raise TypeError(f"layer_names must be a tuple, got {type(layer_names).__name__}")
    
    return {name: is_valid_layer(name) for name in layer_names}


def get_layer_dependencies(layer_name: LayerName) -> Tuple[LayerName, ...]:
    """
    Get the layers that a given layer depends on (lower layers in the architecture).
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple[LayerName, ...]: Tuple of layer names that this layer depends on
    """
    return LAYER_DEPENDENCY_RULES.get(layer_name, ())


def can_access_layer(source_layer: LayerName, target_layer: LayerName) -> bool:
    """
    Check if a source layer can access a target layer (following dependency rules).
    
    Args:
        source_layer: The layer trying to access
        target_layer: The layer being accessed
        
    Returns:
        bool: True if access is allowed, False otherwise
    """
    if source_layer == target_layer:
        return True
    
    dependencies = get_layer_dependencies(source_layer)
    return target_layer in dependencies


def get_layer_order(layer_name: LayerName) -> int:
    """
    Get the hierarchical order of a layer (lower number = deeper layer).
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        int: Order number (1 = domain, 4 = presentation)
    """
    return LAYER_ORDER[layer_name].value


def is_higher_layer(layer1: LayerName, layer2: LayerName) -> bool:
    """
    Check if layer1 is higher in the hierarchy than layer2.
    
    Args:
        layer1: First layer to compare
        layer2: Second layer to compare
        
    Returns:
        bool: True if layer1 is higher than layer2
    """
    return get_layer_order(layer1) > get_layer_order(layer2)


def get_layers_below(layer_name: LayerName) -> Tuple[LayerName, ...]:
    """
    Get all layers that are below (deeper than) the given layer.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple[LayerName, ...]: Tuple of layer names below this layer
    """
    current_order = get_layer_order(layer_name)
    return tuple(
        layer for layer, order in LAYER_ORDER.items()
        if order.value < current_order
    )


def get_layers_above(layer_name: LayerName) -> Tuple[LayerName, ...]:
    """
    Get all layers that are above (higher than) the given layer.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple[LayerName, ...]: Tuple of layer names above this layer
    """
    current_order = get_layer_order(layer_name)
    return tuple(
        layer for layer, order in LAYER_ORDER.items()
        if order.value > current_order
    )


@lru_cache(maxsize=1)
def get_layer_hierarchy() -> List[LayerName]:
    """
    Get layers ordered by hierarchy (from deepest to highest).
    
    Returns:
        List[LayerName]: List of layers ordered from domain to presentation
    """
    return sorted(
        VALID_LAYERS,
        key=get_layer_order
    )


__all__ = [
    "get_layer_info",
    "get_module_info",
    "get_all_layers",
    "get_layer_description",
    "get_layer_count",
    "is_valid_layer",
    "validate_layers",
    "get_layer_dependencies",
    "can_access_layer",
    "get_layer_order",
    "is_higher_layer",
    "get_layers_below",
    "get_layers_above",
    "get_layer_hierarchy",
]

