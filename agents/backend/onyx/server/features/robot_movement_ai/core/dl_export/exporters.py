"""
Model Exporters - Modular Model Export
======================================

Exportadores modulares para diferentes formatos y optimizaciones.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelExporter:
    """Clase base para exportadores."""
    
    def export(self, model: nn.Module, output_path: str, **kwargs):
        """Exportar modelo."""
        raise NotImplementedError


class ONNXExporter(ModelExporter):
    """Exportador a ONNX."""
    
    def export(
        self,
        model: nn.Module,
        output_path: str,
        example_input: torch.Tensor,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Exportar modelo a ONNX.
        
        Args:
            model: Modelo PyTorch
            output_path: Ruta de salida
            example_input: Ejemplo de entrada
            opset_version: Versión de opset
            dynamic_axes: Ejes dinámicos
            **kwargs: Argumentos adicionales
        """
        try:
            import torch.onnx
            
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            torch.onnx.export(
                model,
                example_input,
                output_path,
                opset_version=opset_version,
                input_names=kwargs.get('input_names', ['input']),
                output_names=kwargs.get('output_names', ['output']),
                dynamic_axes=dynamic_axes,
                **{k: v for k, v in kwargs.items() if k not in ['input_names', 'output_names']}
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
        except ImportError:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            raise


class TorchScriptExporter(ModelExporter):
    """Exportador a TorchScript."""
    
    def export(
        self,
        model: nn.Module,
        output_path: str,
        example_input: Optional[torch.Tensor] = None,
        method: str = 'trace',
        **kwargs
    ):
        """
        Exportar modelo a TorchScript.
        
        Args:
            model: Modelo PyTorch
            output_path: Ruta de salida
            example_input: Ejemplo de entrada (para tracing)
            method: Método ('trace' o 'script')
            **kwargs: Argumentos adicionales
        """
        try:
            if method == 'trace':
                if example_input is None:
                    raise ValueError("example_input required for tracing")
                traced_model = torch.jit.trace(model, example_input)
            elif method == 'script':
                traced_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            traced_model.save(output_path)
            logger.info(f"Model exported to TorchScript: {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to TorchScript: {e}")
            raise


class SafetensorsExporter(ModelExporter):
    """Exportador a SafeTensors."""
    
    def export(
        self,
        model: nn.Module,
        output_path: str,
        **kwargs
    ):
        """
        Exportar modelo a SafeTensors.
        
        Args:
            model: Modelo PyTorch
            output_path: Ruta de salida
            **kwargs: Argumentos adicionales
        """
        try:
            from safetensors.torch import save_file
            
            state_dict = model.state_dict()
            save_file(state_dict, output_path)
            
            logger.info(f"Model exported to SafeTensors: {output_path}")
        except ImportError:
            raise ImportError("safetensors not available. Install with: pip install safetensors")
        except Exception as e:
            logger.error(f"Error exporting to SafeTensors: {e}")
            raise


class ExporterFactory:
    """Factory para exportadores."""
    
    _exporters = {
        'onnx': ONNXExporter,
        'torchscript': TorchScriptExporter,
        'safetensors': SafetensorsExporter
    }
    
    @classmethod
    def get_exporter(cls, format_type: str) -> ModelExporter:
        """
        Obtener exportador por formato.
        
        Args:
            format_type: Tipo de formato
            
        Returns:
            Exportador
        """
        if format_type not in cls._exporters:
            raise ValueError(f"Unknown export format: {format_type}")
        
        return cls._exporters[format_type]()
    
    @classmethod
    def register_exporter(cls, format_type: str, exporter_class: type):
        """Registrar nuevo exportador."""
        cls._exporters[format_type] = exporter_class


def export_model(
    model: nn.Module,
    output_path: str,
    format_type: str = 'onnx',
    **kwargs
):
    """
    Exportar modelo a formato específico.
    
    Args:
        model: Modelo PyTorch
        output_path: Ruta de salida
        format_type: Tipo de formato ('onnx', 'torchscript', 'safetensors')
        **kwargs: Argumentos adicionales
    """
    exporter = ExporterFactory.get_exporter(format_type)
    exporter.export(model, output_path, **kwargs)









