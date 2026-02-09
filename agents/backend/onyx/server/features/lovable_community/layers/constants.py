"""
Layer Architecture Constants and Type Definitions

This module defines all constants, type aliases, and enums used throughout
the layer architecture system.
"""

from typing import Dict, Literal, Any
from enum import IntEnum

# Type aliases for better type safety
LayerName = Literal["domain", "application", "infrastructure", "presentation"]
LayerInfo = Dict[str, str]
ModuleInfo = Dict[str, Any]

# Layer names as constants
LAYER_DOMAIN = "domain"
LAYER_APPLICATION = "application"
LAYER_INFRASTRUCTURE = "infrastructure"
LAYER_PRESENTATION = "presentation"

# Layer hierarchy order (lower number = lower layer)
class LayerOrder(IntEnum):
    """Order of layers in the architecture (lower = deeper layer)"""
    DOMAIN = 1
    INFRASTRUCTURE = 2
    APPLICATION = 3
    PRESENTATION = 4

# Layer order mapping
LAYER_ORDER: Dict[LayerName, LayerOrder] = {
    LAYER_DOMAIN: LayerOrder.DOMAIN,
    LAYER_INFRASTRUCTURE: LayerOrder.INFRASTRUCTURE,
    LAYER_APPLICATION: LayerOrder.APPLICATION,
    LAYER_PRESENTATION: LayerOrder.PRESENTATION,
}

# Valid layer names set (for efficient validation)
VALID_LAYERS = frozenset({
    LAYER_DOMAIN,
    LAYER_APPLICATION,
    LAYER_INFRASTRUCTURE,
    LAYER_PRESENTATION,
})

# Layer descriptions mapping
LAYER_DESCRIPTIONS = {
    LAYER_DOMAIN: "Business entities and rules",
    LAYER_APPLICATION: "Use cases and application services",
    LAYER_INFRASTRUCTURE: "Technical implementations",
    LAYER_PRESENTATION: "API layer",
}

# Layer dependency rules (source -> targets)
LAYER_DEPENDENCY_RULES = {
    LAYER_DOMAIN: (),
    LAYER_APPLICATION: (LAYER_DOMAIN,),
    LAYER_INFRASTRUCTURE: (LAYER_DOMAIN,),
    LAYER_PRESENTATION: (LAYER_DOMAIN, LAYER_APPLICATION),
}

__all__ = [
    "LayerName",
    "LayerInfo",
    "ModuleInfo",
    "LAYER_DOMAIN",
    "LAYER_APPLICATION",
    "LAYER_INFRASTRUCTURE",
    "LAYER_PRESENTATION",
    "LayerOrder",
    "LAYER_ORDER",
    "VALID_LAYERS",
    "LAYER_DESCRIPTIONS",
    "LAYER_DEPENDENCY_RULES",
]

