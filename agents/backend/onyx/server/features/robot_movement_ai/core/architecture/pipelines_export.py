"""
Model Export and Optimization Module
=====================================

Sistema profesional para exportar y optimizar modelos.
Soporta ONNX, TorchScript, TensorRT, y optimizaciones avanzadas.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None
    logging.warning("ONNX not available. Install with: pip install onnx onnxruntime")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    logging.warning("TensorRT not available.")

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Exportador profesional de modelos.
    
    Soporta:
    - PyTorch (native)
    - ONNX
    - TorchScript
    - TensorRT (si está disponible)
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar exportador.
        
        Args:
            model: Modelo PyTorch a exportar
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelExporter")
        
        self.model = model
        self.model.eval()
        logger.info("ModelExporter initialized")
    
    def export_pytorch(
        self,
        output_path: str,
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> str:
        """
        Exportar modelo en formato PyTorch nativo.
        
        Args:
            output_path: Ruta de salida
            include_optimizer: Incluir estado del optimizador
            optimizer: Optimizador (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__
        }
        
        if include_optimizer and optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, output_path)
        logger.info(f"PyTorch model exported to {output_path}")
        return output_path
    
    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14,
        do_constant_folding: bool = True
    ) -> str:
        """
        Exportar modelo a ONNX.
        
        Args:
            output_path: Ruta de salida
            input_shape: Forma de entrada (sin batch dimension)
            input_names: Nombres de inputs
            output_names: Nombres de outputs
            dynamic_axes: Ejes dinámicos
            opset_version: Versión de opset ONNX
            do_constant_folding: Hacer constant folding
            
        Returns:
            Ruta del archivo guardado
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required. Install with: pip install onnx")
        
        # Crear dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Exportar
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            verbose=False
        )
        
        # Verificar modelo
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model exported and verified: {output_path}")
        except Exception as e:
            logger.warning(f"ONNX model verification failed: {e}")
        
        return output_path
    
    def export_torchscript(
        self,
        output_path: str,
        method: str = "trace",
        example_input: Optional[torch.Tensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None
    ) -> str:
        """
        Exportar modelo a TorchScript.
        
        Args:
            output_path: Ruta de salida
            method: Método ("trace" o "script")
            example_input: Input de ejemplo (para trace)
            input_shape: Forma de entrada (si no se proporciona example_input)
            
        Returns:
            Ruta del archivo guardado
        """
        if method == "trace":
            if example_input is None:
                if input_shape is None:
                    raise ValueError("Either example_input or input_shape must be provided")
                example_input = torch.randn(1, *input_shape)
            
            traced_model = torch.jit.trace(self.model, example_input)
            traced_model.save(output_path)
            logger.info(f"TorchScript model (traced) exported to {output_path}")
        
        elif method == "script":
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)
            logger.info(f"TorchScript model (scripted) exported to {output_path}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return output_path
    
    def optimize_onnx(
        self,
        onnx_path: str,
        output_path: str,
        optimization_level: str = "all"
    ) -> str:
        """
        Optimizar modelo ONNX.
        
        Args:
            onnx_path: Ruta al modelo ONNX
            output_path: Ruta de salida optimizada
            optimization_level: Nivel de optimización ("basic", "extended", "all")
            
        Returns:
            Ruta del archivo optimizado
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required")
        
        try:
            from onnxruntime.transformers import optimizer
            
            optimization_options = {
                "basic": ["eliminate_nop_transpose", "eliminate_nop_pad"],
                "extended": ["eliminate_nop_transpose", "eliminate_nop_pad", "fuse_matmul_add_bias_into_gemm"],
                "all": None  # Todas las optimizaciones
            }
            
            opt_options = optimization_options.get(optimization_level, None)
            optimized_model = optimizer.optimize_model(onnx_path, optimization_options=opt_options)
            optimized_model.save_model_to_file(output_path)
            
            logger.info(f"ONNX model optimized and saved to {output_path}")
            return output_path
        except ImportError:
            logger.warning("onnxruntime.transformers not available, skipping optimization")
            return onnx_path


class ModelQuantizer:
    """
    Cuantizador de modelos para optimización.
    
    Soporta:
    - Dynamic quantization
    - Static quantization
    - QAT (Quantization Aware Training)
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar cuantizador.
        
        Args:
            model: Modelo PyTorch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        logger.info("ModelQuantizer initialized")
    
    def dynamic_quantize(self) -> nn.Module:
        """
        Cuantización dinámica.
        
        Returns:
            Modelo cuantizado
        """
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        logger.info("Model dynamically quantized")
        return quantized_model
    
    def static_quantize(
        self,
        calibration_data: List[torch.Tensor],
        backend: str = "fbgemm"
    ) -> nn.Module:
        """
        Cuantización estática.
        
        Args:
            calibration_data: Datos de calibración
            backend: Backend de cuantización
            
        Returns:
            Modelo cuantizado
        """
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Preparar modelo
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrar
        with torch.no_grad():
            for data in calibration_data:
                self.model(data)
        
        # Convertir
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        logger.info("Model statically quantized")
        return quantized_model


class ModelPruner:
    """
    Pruner de modelos para compresión.
    
    Soporta diferentes estrategias de pruning.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar pruner.
        
        Args:
            model: Modelo PyTorch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        logger.info("ModelPruner initialized")
    
    def magnitude_pruning(
        self,
        amount: float = 0.2,
        sparsity_type: str = "unstructured"
    ) -> nn.Module:
        """
        Pruning basado en magnitud.
        
        Args:
            amount: Cantidad de pruning (0.0-1.0)
            sparsity_type: Tipo ("unstructured" o "structured")
            
        Returns:
            Modelo podado
        """
        try:
            from torch.nn.utils import prune
            
            parameters_to_prune = [
                (module, 'weight') for module in self.model.modules()
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))
            ]
            
            if sparsity_type == "unstructured":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=amount
                )
            else:
                prune.global_structured(
                    parameters_to_prune,
                    pruning_method=prune.LnStructured,
                    amount=amount,
                    n=2
                )
            
            logger.info(f"Model pruned: {amount*100:.1f}% sparsity ({sparsity_type})")
            return self.model
        except ImportError:
            logger.warning("Pruning utilities not available")
            return self.model

