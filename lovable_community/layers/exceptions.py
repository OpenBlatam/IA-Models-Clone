"""
Layer Architecture Exceptions

Custom exceptions for layer architecture validation and access control.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .constants import LayerName


class LayerAccessError(ValueError):
    """Exception raised when a layer tries to access another layer illegally."""
    
    def __init__(self, source_layer: "LayerName", target_layer: "LayerName"):
        self.source_layer = source_layer
        self.target_layer = target_layer
        message = (
            f"{source_layer.capitalize()} layer cannot access "
            f"{target_layer.capitalize()} layer (violates dependency rule)"
        )
        super().__init__(message)


class InvalidLayerError(ValueError):
    """Exception raised when an invalid layer name is used."""
    
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        from .constants import VALID_LAYERS
        valid_layers = ", ".join(sorted(VALID_LAYERS))
        message = (
            f"Invalid layer name: '{layer_name}'. "
            f"Valid layers are: {valid_layers}"
        )
        super().__init__(message)


__all__ = [
    "LayerAccessError",
    "InvalidLayerError",
]



