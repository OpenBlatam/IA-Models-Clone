"""
Model Export and Deployment
==========================
Export models for production deployment
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from pathlib import Path
import structlog
import json
import onnx
import onnxruntime as ort

try:
    import torch.onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = structlog.get_logger()


class ModelExporter:
    """
    Export models to various formats for deployment
    """
    
    def __init__(self, output_dir: str = "./exports"):
        """
        Initialize exporter
        
        Args:
            output_dir: Output directory for exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelExporter initialized", output_dir=str(self.output_dir))
    
    def export_pytorch(
        self,
        model: nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export PyTorch model
        
        Args:
            model: Model to export
            model_name: Name for the model
            metadata: Additional metadata
            
        Returns:
            Path to exported model
        """
        export_path = self.output_dir / f"{model_name}.pt"
        
        export_data = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "metadata": metadata or {}
        }
        
        torch.save(export_data, export_path)
        
        logger.info("PyTorch model exported", path=str(export_path))
        
        return str(export_path)
    
    def export_onnx(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: tuple,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """
        Export model to ONNX format
        
        Args:
            model: Model to export
            model_name: Name for the model
            input_shape: Input shape (batch_size, seq_len)
            input_names: Input names
            output_names: Output names
            dynamic_axes: Dynamic axes for variable batch/sequence length
            
        Returns:
            Path to exported ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX export requires torch.onnx")
        
        export_path = self.output_dir / f"{model_name}.onnx"
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        if input_names is None:
            input_names = ["input_ids", "attention_mask"]
        if output_names is None:
            output_names = ["output"]
        
        try:
            torch.onnx.export(
                model,
                (dummy_input, torch.ones_like(dummy_input)),  # (input_ids, attention_mask)
                export_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11,
                do_constant_folding=True,
                export_params=True
            )
            
            logger.info("ONNX model exported", path=str(export_path))
            
            # Verify ONNX model
            self._verify_onnx_model(export_path)
            
            return str(export_path)
            
        except Exception as e:
            logger.error("Error exporting ONNX model", error=str(e))
            raise
    
    def export_torchscript(
        self,
        model: nn.Module,
        model_name: str,
        example_input: Optional[torch.Tensor] = None
    ) -> str:
        """
        Export model to TorchScript
        
        Args:
            model: Model to export
            model_name: Name for the model
            example_input: Example input for tracing
            
        Returns:
            Path to exported TorchScript model
        """
        export_path = self.output_dir / f"{model_name}.torchscript"
        
        model.eval()
        
        try:
            if example_input is not None:
                # Trace model
                traced_model = torch.jit.trace(model, example_input)
            else:
                # Script model (more flexible but may not work for all models)
                traced_model = torch.jit.script(model)
            
            traced_model.save(str(export_path))
            
            logger.info("TorchScript model exported", path=str(export_path))
            
            return str(export_path)
            
        except Exception as e:
            logger.error("Error exporting TorchScript model", error=str(e))
            raise
    
    def export_metadata(
        self,
        model_name: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Export model metadata
        
        Args:
            model_name: Model name
            metadata: Metadata dictionary
            
        Returns:
            Path to metadata file
        """
        metadata_path = self.output_dir / f"{model_name}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Metadata exported", path=str(metadata_path))
        
        return str(metadata_path)
    
    def _verify_onnx_model(self, onnx_path: Path) -> None:
        """Verify ONNX model"""
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verified")
        except Exception as e:
            logger.warning("ONNX model verification failed", error=str(e))


class ModelLoader:
    """Load exported models"""
    
    @staticmethod
    def load_pytorch(
        model_path: str,
        model_class: Optional[type] = None
    ) -> nn.Module:
        """
        Load PyTorch model
        
        Args:
            model_path: Path to model file
            model_class: Model class (optional)
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if model_class:
            model = model_class()
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError("model_class required for loading")
        
        return model
    
    @staticmethod
    def load_onnx(
        onnx_path: str,
        providers: Optional[List[str]] = None
    ) -> ort.InferenceSession:
        """
        Load ONNX model for inference
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers (e.g., ['CPUExecutionProvider', 'CUDAExecutionProvider'])
            
        Returns:
            ONNX Runtime inference session
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )
        
        logger.info("ONNX model loaded", providers=providers)
        
        return session


# Global exporter instance
model_exporter = ModelExporter()




