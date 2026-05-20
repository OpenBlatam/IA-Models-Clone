"""
Model Composition System
Build complex models from simple components
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
    Composes models from modular components
    Follows builder pattern for flexible model construction
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
        Add a component to the composition
        
        Args:
            name: Component name
            component: Module component
            is_input: Whether this is an input component
            is_output: Whether this is an output component
        
        Returns:
            Self for chaining
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
        Connect two components
        
        Args:
            from_component: Source component name
            to_component: Target component name
            transform: Optional transformation function
        
        Returns:
            Self for chaining
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
        Build the composed model
        
        Returns:
            Composed model module
        """
        if self.input_name is None:
            raise ValueError("No input component specified")
        if self.output_name is None:
            raise ValueError("No output component specified")
        
        return ComposedModel(
            components=self.components,
            connections=self.connections,
            input_name=self.input_name,
            output_name=self.output_name
        )


class ComposedModel(nn.Module):
    """
    Composed model from multiple components
    """
    
    def __init__(
        self,
        components: Dict[str, nn.Module],
        connections: List[Dict[str, Any]],
        input_name: str,
        output_name: str
    ):
        super().__init__()
        self.components = nn.ModuleDict(components)
        self.connections = connections
        self.input_name = input_name
        self.output_name = output_name
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through composed model"""
        # Track outputs of each component
        outputs = {self.input_name: x}
        
        # Process connections in order
        for connection in self.connections:
            from_name = connection["from"]
            to_name = connection["to"]
            transform = connection.get("transform")
            
            # Get input
            input_data = outputs[from_name]
            
            # Apply transformation if provided
            if transform:
                input_data = transform(input_data)
            
            # Forward through component
            component = self.components[to_name]
            output = component(input_data)
            outputs[to_name] = output
        
        # Return final output
        return outputs[self.output_name]


class SequentialComposer:
    """
    Simplified composer for sequential models
    """
    
    def __init__(self):
        self.layers: List[nn.Module] = []
    
    def add(self, layer: nn.Module) -> 'SequentialComposer':
        """Add a layer"""
        self.layers.append(layer)
        return self
    
    def build(self) -> nn.Sequential:
        """Build sequential model"""
        return nn.Sequential(*self.layers)


class ParallelComposer:
    """
    Composer for parallel processing branches
    """
    
    def __init__(self):
        self.branches: Dict[str, nn.Module] = {}
        self.merge_strategy: str = "concat"  # "concat", "add", "multiply"
    
    def add_branch(self, name: str, branch: nn.Module) -> 'ParallelComposer':
        """Add a parallel branch"""
        self.branches[name] = branch
        return self
    
    def set_merge_strategy(self, strategy: str) -> 'ParallelComposer':
        """Set merge strategy"""
        self.merge_strategy = strategy
        return self
    
    def build(self) -> nn.Module:
        """Build parallel model"""
        return ParallelModel(
            branches=self.branches,
            merge_strategy=self.merge_strategy
        )


class ParallelModel(nn.Module):
    """Model with parallel branches"""
    
    def __init__(self, branches: Dict[str, nn.Module], merge_strategy: str = "concat"):
        super().__init__()
        self.branches = nn.ModuleDict(branches)
        self.merge_strategy = merge_strategy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through parallel branches"""
        outputs = {}
        for name, branch in self.branches.items():
            outputs[name] = branch(x)
        
        # Merge outputs
        if self.merge_strategy == "concat":
            return torch.cat(list(outputs.values()), dim=-1)
        elif self.merge_strategy == "add":
            return sum(outputs.values())
        elif self.merge_strategy == "multiply":
            result = list(outputs.values())[0]
            for output in list(outputs.values())[1:]:
                result = result * output
            return result
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")



