"""
Model Export/Import System - Sistema de exportación e importación de modelos
=============================================================================
"""

import logging
import torch
import torch.nn as nn
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Formatos de exportación"""
    PYTORCH = "pytorch"  # .pt or .pth
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORRT = "tensorrt"  # Requires TensorRT
    OPENVINO = "openvino"  # Requires OpenVINO


@dataclass
class ExportConfig:
    """Configuración de exportación"""
    format: ExportFormat = ExportFormat.PYTORCH
    input_shape: Optional[Tuple[int, ...]] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    dynamic_axes: Optional[Dict[str, Any]] = None
    opset_version: int = 11
    do_constant_folding: bool = True
    export_params: bool = True
    verbose: bool = False


class ModelExporter:
    """Exportador de modelos"""
    
    def __init__(self):
        self.export_history: List[Dict[str, Any]] = []
    
    def export_model(
        self,
        model: nn.Module,
        save_path: str,
        config: ExportConfig,
        example_input: Optional[torch.Tensor] = None
    ) -> str:
        """Exporta un modelo"""
        model.eval()
        
        if config.format == ExportFormat.PYTORCH:
            return self._export_pytorch(model, save_path)
        elif config.format == ExportFormat.ONNX:
            return self._export_onnx(model, save_path, config, example_input)
        elif config.format == ExportFormat.TORCHSCRIPT:
            return self._export_torchscript(model, save_path, example_input)
        else:
            raise ValueError(f"Formato {config.format} no soportado aún")
    
    def _export_pytorch(self, model: nn.Module, save_path: str) -> str:
        """Exporta a formato PyTorch"""
        torch.save(model.state_dict(), save_path)
        logger.info(f"Modelo exportado a PyTorch: {save_path}")
        return save_path
    
    def _export_onnx(
        self,
        model: nn.Module,
        save_path: str,
        config: ExportConfig,
        example_input: Optional[torch.Tensor]
    ) -> str:
        """Exporta a formato ONNX"""
        try:
            import torch.onnx
            
            if example_input is None:
                if config.input_shape is None:
                    raise ValueError("Se requiere example_input o input_shape para exportar a ONNX")
                example_input = torch.randn(config.input_shape)
            
            torch.onnx.export(
                model,
                example_input,
                save_path,
                input_names=config.input_names or ["input"],
                output_names=config.output_names or ["output"],
                dynamic_axes=config.dynamic_axes,
                opset_version=config.opset_version,
                do_constant_folding=config.do_constant_folding,
                export_params=config.export_params,
                verbose=config.verbose
            )
            
            logger.info(f"Modelo exportado a ONNX: {save_path}")
            return save_path
        except ImportError:
            raise ImportError("torch.onnx no disponible")
    
    def _export_torchscript(
        self,
        model: nn.Module,
        save_path: str,
        example_input: Optional[torch.Tensor]
    ) -> str:
        """Exporta a TorchScript"""
        if example_input is None:
            raise ValueError("Se requiere example_input para exportar a TorchScript")
        
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(save_path)
        
        logger.info(f"Modelo exportado a TorchScript: {save_path}")
        return save_path
    
    def load_model(
        self,
        model_path: str,
        model_class: Optional[type] = None,
        map_location: str = "cpu"
    ) -> nn.Module:
        """Carga un modelo"""
        if model_path.endswith('.onnx'):
            return self._load_onnx(model_path)
        elif model_path.endswith('.pt') or model_path.endswith('.pth'):
            if model_class:
                model = model_class()
                model.load_state_dict(torch.load(model_path, map_location=map_location))
                return model
            else:
                return torch.load(model_path, map_location=map_location)
        elif model_path.endswith('.ts'):
            return torch.jit.load(model_path, map_location=map_location)
        else:
            raise ValueError(f"Formato de archivo no reconocido: {model_path}")
    
    def _load_onnx(self, model_path: str) -> Any:
        """Carga modelo ONNX"""
        try:
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        except ImportError:
            logger.warning("onnxruntime no disponible, no se puede cargar modelo ONNX")
            return None
    
    def verify_export(
        self,
        original_model: nn.Module,
        exported_path: str,
        example_input: torch.Tensor,
        rtol: float = 1e-3,
        atol: float = 1e-5
    ) -> bool:
        """Verifica que el modelo exportado produce los mismos resultados"""
        original_model.eval()
        
        with torch.no_grad():
            original_output = original_model(example_input)
        
        # Cargar modelo exportado
        if exported_path.endswith('.onnx'):
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(exported_path)
                input_name = session.get_inputs()[0].name
                exported_output = session.run(None, {input_name: example_input.numpy()})
                exported_output = torch.tensor(exported_output[0])
            except ImportError:
                logger.warning("No se puede verificar ONNX sin onnxruntime")
                return False
        elif exported_path.endswith('.ts'):
            exported_model = torch.jit.load(exported_path)
            exported_model.eval()
            with torch.no_grad():
                exported_output = exported_model(example_input)
        else:
            logger.warning("Verificación solo disponible para ONNX y TorchScript")
            return False
        
        # Comparar outputs
        if torch.allclose(original_output, exported_output, rtol=rtol, atol=atol):
            logger.info("Modelo exportado verificado correctamente")
            return True
        else:
            logger.warning("Modelo exportado no coincide con el original")
            return False




