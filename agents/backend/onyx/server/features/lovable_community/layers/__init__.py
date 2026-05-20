"""
Layered Architecture Module

Clean Architecture with clear separation of concerns:
- Domain: Business entities and rules
- Application: Use cases and application services
- Infrastructure: Technical implementations
- Presentation: API layer

This module provides a comprehensive interface to all layers of the architecture,
including:
- Domain entities, exceptions, and interfaces
- Application services
- Architecture utilities organized in modular sub-packages
- Type-safe layer management
- Dependency validation and cycle detection
- Complexity and impact analysis
- Health monitoring and metrics

Example usage:
    # Import domain entities and services
    from layers import (
        PublishedChat,
        ChatService,
        ChatNotFoundError,
        IChatRepository,
    )
    
    # Use architecture utilities
    from layers import (
        get_layer_info,
        validate_architecture,
        get_architecture_health,
        create_architecture_diagram,
        export_architecture_json,
    )
    
    # Validate architecture
    is_valid, issues = validate_architecture()
    
    # Get health metrics
    health = get_architecture_health()
    
    # Create visual diagram
    diagram = create_architecture_diagram()
    
    # Export to JSON
    json_data = export_architecture_json()

Version: 1.0.0
Author: Lovable Community Team
"""

__version__ = "1.0.0"
__author__ = "Lovable Community Team"
__description__ = "Clean Architecture implementation with layered separation"

# Import constants and types
from .constants import (
    LayerName,
    LayerInfo,
    ModuleInfo,
    LAYER_DOMAIN,
    LAYER_APPLICATION,
    LAYER_INFRASTRUCTURE,
    LAYER_PRESENTATION,
    LayerOrder,
    LAYER_ORDER,
    VALID_LAYERS,
    LAYER_DESCRIPTIONS,
    LAYER_DEPENDENCY_RULES,
)

# Import exceptions
from .exceptions import (
    LayerAccessError,
    InvalidLayerError,
)

# Import utilities
from .utils import (
    get_layer_info,
    get_module_info,
    get_all_layers,
    get_layer_description,
    get_layer_count,
    is_valid_layer,
    validate_layers,
    get_layer_dependencies,
    can_access_layer,
    get_layer_order,
    is_higher_layer,
    get_layers_below,
    get_layers_above,
    get_layer_hierarchy,
)

# Import validators
from .validators import (
    validate_layer_access,
    require_layer_access,
    require_valid_layer,
    get_layer_dependency_graph,
    check_dependency_cycle,
    validate_layer_chain,
    validate_architecture,
)

# Import analysis functions
from .analysis import (
    find_accessible_layers,
    find_layers_that_can_access,
    get_transitive_dependencies,
    get_layer_path,
    get_layer_distance,
    get_layer_ancestors,
    get_dependent_layers,
    get_layer_descendants,
    get_layer_relationships,
    get_layer_depth,
    get_nearest_common_ancestor,
    layer_to_dict,
    get_all_layers_info,
    get_all_layer_dependencies,
)

# Import metrics
from .metrics import (
    get_layer_complexity_score,
    get_layer_impact_score,
    get_layer_cohesion_score,
    calculate_layer_coupling,
    count_layer_connections,
    get_all_layer_metrics,
    get_most_complex_layers,
    get_least_complex_layers,
    get_high_impact_layers,
    get_layer_statistics,
)

# Import domain and application layers
from .application import (
    ChatService,
    RankingService,
)

from .domain import (
    ChatAIMetadata,
    ChatEmbedding,
    ChatRemix,
    ChatVote,
    ChatView,
    ChatNotFoundError,
    DatabaseError,
    DuplicateVoteError,
    IAIProcessor,
    IChatRepository,
    IRemixRepository,
    IRankingService,
    IScoreManager,
    IValidator,
    IVoteRepository,
    IViewRepository,
    InvalidChatError,
    PublishedChat,
    RemixError,
)

# Re-export additional utility functions for backward compatibility
# These are convenience functions that combine utilities from different modules
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any, Callable

# Additional convenience functions
def get_layer_by_order(order: int) -> Optional[LayerName]:
    """Get layer name by its hierarchical order."""
    for layer, layer_order in LAYER_ORDER.items():
        if layer_order.value == order:
            return layer
    return None


def get_next_layer(layer_name: LayerName) -> Optional[LayerName]:
    """Get the next layer in hierarchy order."""
    current_order = get_layer_order(layer_name)
    next_order = current_order + 1
    return get_layer_by_order(next_order)


def get_previous_layer(layer_name: LayerName) -> Optional[LayerName]:
    """Get the previous layer in hierarchy order."""
    current_order = get_layer_order(layer_name)
    prev_order = current_order - 1
    return get_layer_by_order(prev_order)


def is_adjacent_layer(layer1: LayerName, layer2: LayerName) -> bool:
    """Check if two layers are adjacent in the hierarchy."""
    order1 = get_layer_order(layer1)
    order2 = get_layer_order(layer2)
    return abs(order1 - order2) == 1


def get_layer_neighbors(layer_name: LayerName) -> Tuple[Optional[LayerName], Optional[LayerName]]:
    """Get the previous and next layers in hierarchy."""
    return (get_previous_layer(layer_name), get_next_layer(layer_name))


def is_layer_independent(layer_name: LayerName) -> bool:
    """Check if a layer has no dependencies (is independent)."""
    return len(get_layer_dependencies(layer_name)) == 0


def get_leaf_layers() -> Tuple[LayerName, ...]:
    """Get layers that have no dependencies (leaf nodes in dependency graph)."""
    return tuple(
        layer for layer in VALID_LAYERS
        if len(get_layer_dependencies(layer)) == 0
    )


def get_root_layers() -> Tuple[LayerName, ...]:
    """Get layers that are not depended upon by any other layer (root nodes)."""
    all_deps = get_all_layer_dependencies()
    all_depended_upon = set()
    for deps in all_deps.values():
        all_depended_upon.update(deps)
    
    return tuple(
        layer for layer in VALID_LAYERS
        if layer not in all_depended_upon
    )


def compare_layers(layer1: LayerName, layer2: LayerName) -> Dict[str, Any]:
    """Compare two layers and return detailed comparison."""
    return {
        "layer1_higher": is_higher_layer(layer1, layer2),
        "layer2_higher": is_higher_layer(layer2, layer1),
        "can_layer1_access_layer2": can_access_layer(layer1, layer2),
        "can_layer2_access_layer1": can_access_layer(layer2, layer1),
        "order_difference": abs(get_layer_order(layer1) - get_layer_order(layer2)),
        "layer1_info": layer_to_dict(layer1),
        "layer2_info": layer_to_dict(layer2),
    }


@lru_cache(maxsize=1)
def get_architecture_health() -> Dict[str, Any]:
    """
    Get overall health metrics of the architecture.
    
    Returns:
        Dict[str, Any]: Health metrics
    """
    is_valid, issues = validate_architecture()
    stats = get_layer_statistics()
    
    avg_complexity = sum(
        get_layer_complexity_score(layer)
        for layer in VALID_LAYERS
    ) / get_layer_count()
    
    max_complexity = max(
        get_layer_complexity_score(layer)
        for layer in VALID_LAYERS
    )
    
    return {
        "is_healthy": is_valid,
        "issues": issues,
        "has_cycles": stats["has_cycles"],
        "average_complexity": avg_complexity,
        "max_complexity": max_complexity,
        "total_layers": stats["total_layers"],
        "max_depth": stats["max_dependency_depth"],
        "health_score": 100.0 if is_valid and not stats["has_cycles"] else 50.0,
    }


def export_architecture_json() -> str:
    """
    Export the entire architecture as JSON string.
    
    Returns:
        str: JSON string representation of the architecture
    """
    import json
    architecture_data = {
        "version": __version__,
        "metadata": get_module_info(),
        "layers": get_all_layers_info(),
        "statistics": get_layer_statistics(),
        "health": get_architecture_health(),
        "dependency_graph": {
            layer: list(deps)
            for layer, deps in get_all_layer_dependencies().items()
        },
        "hierarchy": get_layer_hierarchy(),
    }
    return json.dumps(architecture_data, indent=2, ensure_ascii=False)


def format_layer_summary(layer_name: LayerName) -> str:
    """Format a human-readable summary of a layer."""
    info = layer_to_dict(layer_name)
    deps_str = ", ".join(info["dependencies"]) if info["dependencies"] else "none"
    accessible_str = ", ".join(info["accessible_layers"])
    
    return (
        f"Layer: {info['name']} (Order: {info['order']})\n"
        f"Description: {info['description']}\n"
        f"Dependencies: {deps_str}\n"
        f"Can access: {accessible_str}\n"
        f"Layers below: {', '.join(info['layers_below']) if info['layers_below'] else 'none'}\n"
        f"Layers above: {', '.join(info['layers_above']) if info['layers_above'] else 'none'}"
    )


@lru_cache(maxsize=1)
def get_architecture_summary() -> str:
    """Get a formatted summary of the entire architecture."""
    hierarchy = get_layer_hierarchy()
    stats = get_layer_statistics()
    cycle = check_dependency_cycle()
    
    summary = f"""Architecture Summary:
Total Layers: {stats['total_layers']}
Hierarchy: {' -> '.join(hierarchy)}
Max Dependency Depth: {stats['max_dependency_depth']}
Total Dependencies: {stats['total_dependencies']}
Has Cycles: {cycle is not None}
"""
    
    if cycle:
        summary += f"Cycle Detected: {' -> '.join(cycle)}\n"
    
    summary += "\nLayer Details:\n"
    for layer in hierarchy:
        summary += f"\n{format_layer_summary(layer)}\n"
    
    return summary


def create_architecture_diagram(format: str = "text") -> str:
    """
    Create a visual diagram of the architecture.
    
    Args:
        format: Output format ("text" or "ascii")
        
    Returns:
        str: Formatted diagram
    """
    if format == "text":
        lines = []
        hierarchy = get_layer_hierarchy()
        
        for layer in reversed(hierarchy):
            deps = get_layer_dependencies(layer)
            order = get_layer_order(layer)
            
            if not deps:
                lines.append(f"{layer.capitalize()} Layer ({order})")
            else:
                prefix = "└── " if layer == hierarchy[-1] else "├── "
                lines.append(f"{prefix}{layer.capitalize()} Layer ({order})")
                for i, dep in enumerate(deps):
                    is_last = i == len(deps) - 1
                    connector = "    └── " if is_last else "    ├── "
                    dep_order = get_layer_order(dep)
                    lines.append(f"{connector}{dep.capitalize()} Layer ({dep_order})")
        
        return "\n".join(reversed(lines))
    else:
        return create_architecture_diagram("text")


# Export all public API
__all__ = [
    # Module metadata
    "__version__",
    "__author__",
    "__description__",
    # Type aliases
    "LayerName",
    "LayerInfo",
    "ModuleInfo",
    # Layer constants
    "LAYER_DOMAIN",
    "LAYER_APPLICATION",
    "LAYER_INFRASTRUCTURE",
    "LAYER_PRESENTATION",
    # Utility functions
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
    # Validators
    "validate_layer_access",
    "require_layer_access",
    "require_valid_layer",
    "get_layer_dependency_graph",
    "check_dependency_cycle",
    "validate_layer_chain",
    "validate_architecture",
    # Analysis
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
    # Metrics
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
    # Convenience functions
    "get_layer_by_order",
    "get_next_layer",
    "get_previous_layer",
    "is_adjacent_layer",
    "get_layer_neighbors",
    "is_layer_independent",
    "get_leaf_layers",
    "get_root_layers",
    "compare_layers",
    "get_architecture_health",
    "export_architecture_json",
    "format_layer_summary",
    "get_architecture_summary",
    "create_architecture_diagram",
    # Exceptions
    "LayerAccessError",
    "InvalidLayerError",
    # Enums
    "LayerOrder",
    # Domain Entities
    "ChatAIMetadata",
    "ChatEmbedding",
    "ChatRemix",
    "ChatVote",
    "ChatView",
    "PublishedChat",
    # Domain Exceptions
    "ChatNotFoundError",
    "DatabaseError",
    "DuplicateVoteError",
    "InvalidChatError",
    "RemixError",
    # Domain Interfaces
    "IAIProcessor",
    "IChatRepository",
    "IRemixRepository",
    "IRankingService",
    "IScoreManager",
    "IValidator",
    "IVoteRepository",
    "IViewRepository",
    # Application Services
    "ChatService",
    "RankingService",
]
