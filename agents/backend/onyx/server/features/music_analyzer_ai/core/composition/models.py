"""
Composed Models Module

Implements composed model architectures.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ComposedModel(nn.Module):
    """
    Composed model from multiple components.
    
    Args:
        components: Dictionary of component modules.
        connections: List of connection specifications.
        input_name: Name of input component.
        output_name: Name of output component.
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
        """Forward pass through composed model."""
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


class ParallelModel(nn.Module):
    """
    Model with parallel branches.
    
    Args:
        branches: Dictionary of branch modules.
        merge_strategy: Merge strategy ("concat", "add", "multiply").
    """
    
    def __init__(self, branches: Dict[str, nn.Module], merge_strategy: str = "concat"):
        super().__init__()
        self.branches = nn.ModuleDict(branches)
        self.merge_strategy = merge_strategy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through parallel branches."""
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



