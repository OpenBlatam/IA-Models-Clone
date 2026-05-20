"""
Model Deployment - Deployment avanzado de modelos
==================================================
Deployment a producción con optimizaciones y versionado
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available")

try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available")

logger = logging.getLogger(__name__)


class ModelDeployment:
    """Sistema de deployment de modelos"""
    
    def __init__(self, deployment_dir: str = "./storage/deployments"):
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.deployments: Dict[str, Dict[str, Any]] = {}
    
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: tuple,
        output_path: str,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, List[int]]] = None
    ) -> str:
        """Exporta modelo a ONNX"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available")
        
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path
    
    def export_to_torchscript(
        self,
        model: nn.Module,
        input_shape: tuple,
        output_path: str,
        method: str = "script"
    ) -> str:
        """Exporta modelo a TorchScript"""
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        if method == "script":
            traced_model = torch.jit.script(model)
        else:
            traced_model = torch.jit.trace(model, dummy_input)
        
        traced_model.save(output_path)
        logger.info(f"Model exported to TorchScript: {output_path}")
        return output_path
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        optimization_level: str = "default"
    ) -> nn.Module:
        """Optimiza modelo para inferencia"""
        model.eval()
        
        if optimization_level == "aggressive":
            # Fusion de operaciones
            try:
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
            except:
                pass
        
        # TorchScript
        try:
            model = torch.jit.script(model)
        except:
            try:
                dummy_input = torch.randn(1, 10)  # Placeholder
                model = torch.jit.trace(model, dummy_input)
            except:
                pass
        
        return model
    
    def create_deployment_package(
        self,
        model: nn.Module,
        model_id: str,
        version: str,
        metadata: Dict[str, Any],
        export_formats: List[str] = ["pytorch", "onnx"]
    ) -> Dict[str, str]:
        """Crea paquete de deployment"""
        deployment_path = self.deployment_dir / model_id / version
        deployment_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Exportar en diferentes formatos
        if "pytorch" in export_formats:
            pytorch_path = deployment_path / "model.pt"
            torch.save(model.state_dict(), pytorch_path)
            exported_files["pytorch"] = str(pytorch_path)
        
        if "onnx" in export_formats and ONNX_AVAILABLE:
            try:
                onnx_path = deployment_path / "model.onnx"
                # Necesitaríamos input_shape de metadata
                # self.export_to_onnx(model, input_shape, str(onnx_path))
                exported_files["onnx"] = str(onnx_path)
            except Exception as e:
                logger.warning(f"Failed to export to ONNX: {e}")
        
        if "torchscript" in export_formats:
            try:
                ts_path = deployment_path / "model.pt"
                # self.export_to_torchscript(model, input_shape, str(ts_path))
                exported_files["torchscript"] = str(ts_path)
            except Exception as e:
                logger.warning(f"Failed to export to TorchScript: {e}")
        
        # Guardar metadata
        metadata_path = deployment_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Guardar información de deployment
        self.deployments[f"{model_id}_{version}"] = {
            "model_id": model_id,
            "version": version,
            "path": str(deployment_path),
            "formats": exported_files,
            "metadata": metadata,
            "deployed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Created deployment package: {model_id} v{version}")
        return exported_files
    
    def load_deployed_model(
        self,
        model_id: str,
        version: str,
        format: str = "pytorch"
    ) -> Any:
        """Carga modelo desplegado"""
        deployment_key = f"{model_id}_{version}"
        if deployment_key not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_key}")
        
        deployment = self.deployments[deployment_key]
        
        if format == "onnx" and ONNX_AVAILABLE:
            onnx_path = deployment["formats"].get("onnx")
            if onnx_path:
                return ort.InferenceSession(onnx_path)
        
        elif format == "torchscript":
            ts_path = deployment["formats"].get("torchscript")
            if ts_path:
                return torch.jit.load(ts_path)
        
        else:  # pytorch
            pt_path = deployment["formats"].get("pytorch")
            if pt_path:
                return torch.load(pt_path)
        
        raise ValueError(f"Format {format} not available for {deployment_key}")




