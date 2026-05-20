"""
Model Composer Module

Composes models from modular components using builder pattern.
"""

from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ModelComposer:
    """
    Composes models from modular components.
    Follows builder pattern for flexible model construction.
    """
    
    def __init__(self):
        self.components: Dict[str, nn.Module] = {}
        self.connections: List[Dict[str, Any]] = []
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
    
    def add_component(
        self,
        name: str,
        component: nn.Module,
        is_input: bool = False,
        is_output: bool = False
    ) -> 'ModelComposer':
        """
        Add a component to the composition.
        
        Args:
            name: Component name.
            component: Module component.
            is_input: Whether this is an input component.
            is_output: Whether this is an output component.
        
        Returns:
            Self for chaining.
        """
        self.components[name] = component
        
        if is_input:
            self.input_name = name
        if is_output:
            self.output_name = name
        
        return self
    
    def connect(
        self,
        from_component: str,
        to_component: str,
        transform: Optional[Callable] = None
    ) -> 'ModelComposer':
        """
        Connect two components.
        
        Args:
            from_component: Source component name.
            to_component: Target component name.
            transform: Optional transformation function.
        
        Returns:
            Self for chaining.
        """
        if from_component not in self.components:
            raise ValueError(f"Component {from_component} not found")
        if to_component not in self.components:
            raise ValueError(f"Component {to_component} not found")
        
        self.connections.append({
            "from": from_component,
            "to": to_component,
            "transform": transform
        })
        
        return self
    
    def build(self) -> nn.Module:
        """
        Build the composed model.
        
        Returns:
            Composed model module.
        """
        if self.input_name is None:
            raise ValueError("No input component specified")
        if self.output_name is None:
            raise ValueError("No output component specified")
        
        from .models import ComposedModel
        
        return ComposedModel(
            components=self.components,
            connections=self.connections,
            input_name=self.input_name,
            output_name=self.output_name
        )



