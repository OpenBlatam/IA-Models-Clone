"""
Model Documentation Generator - Generador de documentación de modelos
=======================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelDocumentation:
    """Documentación de modelo"""
    model_name: str
    architecture: str
    total_parameters: int
    trainable_parameters: int
    layers: List[Dict[str, Any]] = field(default_factory=list)
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "layers": self.layers,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "hyperparameters": self.hyperparameters
        }


class ModelDocumentationGenerator:
    """Generador de documentación de modelos"""
    
    def __init__(self, output_dir: str = "./documentation"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_documentation(
        self,
        model: nn.Module,
        model_name: str,
        example_input: Optional[torch.Tensor] = None,
        description: str = "",
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> ModelDocumentation:
        """Genera documentación del modelo"""
        # Analizar modelo
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Analizar capas
        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Solo hojas
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters())
                }
                
                # Información específica por tipo
                if isinstance(module, nn.Linear):
                    layer_info["in_features"] = module.in_features
                    layer_info["out_features"] = module.out_features
                elif isinstance(module, nn.Conv2d):
                    layer_info["in_channels"] = module.in_channels
                    layer_info["out_channels"] = module.out_channels
                    layer_info["kernel_size"] = module.kernel_size
                
                layers.append(layer_info)
        
        # Obtener shapes si hay example_input
        input_shape = None
        output_shape = None
        if example_input is not None:
            input_shape = tuple(example_input.shape)
            model.eval()
            with torch.no_grad():
                output = model(example_input)
                if hasattr(output, 'logits'):
                    output_shape = tuple(output.logits.shape)
                elif hasattr(output, 'shape'):
                    output_shape = tuple(output.shape)
        
        doc = ModelDocumentation(
            model_name=model_name,
            architecture=type(model).__name__,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            layers=layers,
            input_shape=input_shape,
            output_shape=output_shape,
            description=description,
            hyperparameters=hyperparameters or {}
        )
        
        return doc
    
    def save_documentation(
        self,
        documentation: ModelDocumentation,
        format: str = "json"
    ) -> str:
        """Guarda documentación"""
        import os
        
        if format == "json":
            filepath = os.path.join(self.output_dir, f"{documentation.model_name}_doc.json")
            with open(filepath, 'w') as f:
                json.dump(documentation.to_dict(), f, indent=2)
        elif format == "markdown":
            filepath = os.path.join(self.output_dir, f"{documentation.model_name}_doc.md")
            self._save_markdown(documentation, filepath)
        else:
            raise ValueError(f"Formato {format} no soportado")
        
        logger.info(f"Documentación guardada: {filepath}")
        return filepath
    
    def _save_markdown(self, doc: ModelDocumentation, filepath: str):
        """Guarda en formato Markdown"""
        with open(filepath, 'w') as f:
            f.write(f"# {doc.model_name}\n\n")
            f.write(f"**Architecture**: {doc.architecture}\n\n")
            f.write(f"**Description**: {doc.description}\n\n")
            f.write(f"## Parameters\n\n")
            f.write(f"- Total: {doc.total_parameters:,}\n")
            f.write(f"- Trainable: {doc.trainable_parameters:,}\n\n")
            f.write(f"## Layers\n\n")
            for layer in doc.layers:
                f.write(f"- **{layer['name']}** ({layer['type']}): {layer['parameters']:,} params\n")
            if doc.hyperparameters:
                f.write(f"\n## Hyperparameters\n\n")
                for key, value in doc.hyperparameters.items():
                    f.write(f"- {key}: {value}\n")

