"""
Model Migration System
Migrate models between versions and formats
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelMigrator:
    """
    Migrate models between versions and formats
    """
    
    def __init__(self):
        self.migration_scripts: Dict[str, Callable] = {}
    
    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_func: Callable
    ):
        """Register a migration script"""
        key = f"{from_version}->{to_version}"
        self.migration_scripts[key] = migration_func
        logger.info(f"Registered migration: {key}")
    
    def migrate_model(
        self,
        model_path: str,
        from_version: str,
        to_version: str,
        output_path: Optional[str] = None
    ) -> str:
        """Migrate model from one version to another"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for migration")
        
        key = f"{from_version}->{to_version}"
        if key not in self.migration_scripts:
            raise ValueError(f"Migration not found: {key}")
        
        # Load model
        model = torch.load(model_path, map_location="cpu")
        
        # Run migration
        migrated_model = self.migration_scripts[key](model)
        
        # Save migrated model
        if output_path is None:
            output_path = model_path.replace(".pt", f"_v{to_version}.pt")
        
        torch.save(migrated_model, output_path)
        logger.info(f"Migrated model from {from_version} to {to_version}")
        
        return output_path
    
    def convert_to_onnx(
        self,
        model: Any,
        input_shape: tuple,
        output_path: str
    ) -> str:
        """Convert PyTorch model to ONNX"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        try:
            import torch.onnx
            dummy_input = torch.randn(*input_shape)
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            
            logger.info(f"Converted model to ONNX: {output_path}")
            return output_path
        except ImportError:
            raise ImportError("ONNX export requires torch.onnx")
    
    def convert_to_torchscript(
        self,
        model: Any,
        input_shape: tuple,
        output_path: str,
        method: str = "trace"  # "trace" or "script"
    ) -> str:
        """Convert model to TorchScript"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        if method == "trace":
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(output_path)
        elif method == "script":
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Converted model to TorchScript: {output_path}")
        return output_path

