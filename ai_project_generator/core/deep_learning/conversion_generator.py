"""
Conversion Generator - Generador de utilidades de conversión de modelos
=========================================================================

Genera utilidades para convertir modelos a diferentes formatos:
- ONNX export
- TensorRT optimization
- TorchScript export
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConversionGenerator:
    """Generador de utilidades de conversión"""
    
    def __init__(self):
        """Inicializa el generador de conversión"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de conversión.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        conversion_dir = utils_dir / "conversion"
        conversion_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_onnx_converter(conversion_dir, keywords, project_info)
        self._generate_conversion_init(conversion_dir, keywords)
    
    def _generate_conversion_init(
        self,
        conversion_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de conversión"""
        
        init_content = '''"""
Model Conversion Utilities
===========================

Utilidades para convertir modelos a diferentes formatos.
"""

from .onnx_converter import (
    export_to_onnx,
    optimize_onnx_model,
    validate_onnx_model,
)

__all__ = [
    "export_to_onnx",
    "optimize_onnx_model",
    "validate_onnx_model",
]
'''
        
        (conversion_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_onnx_converter(
        self,
        conversion_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de conversión ONNX"""
        
        onnx_content = '''"""
ONNX Converter - Utilidades para conversión a ONNX
====================================================

Herramientas para exportar y optimizar modelos en formato ONNX.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX no disponible. Instala con: pip install onnx onnxruntime")

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...],
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    device: str = "cpu",
) -> Path:
    """
    Exporta modelo PyTorch a ONNX.
    
    Args:
        model: Modelo a exportar
        output_path: Ruta donde guardar el modelo ONNX
        input_shape: Shape del input (sin batch dimension)
        input_names: Nombres de inputs (opcional)
        output_names: Nombres de outputs (opcional)
        opset_version: Versión de opset ONNX
        dynamic_axes: Ejes dinámicos (opcional)
        device: Dispositivo a usar
    
    Returns:
        Ruta del modelo exportado
    
    Raises:
        ImportError: Si ONNX no está disponible
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX no está disponible. Instala con: pip install onnx onnxruntime")
    
    model.eval()
    model.to(device)
    
    # Crear dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Nombres por defecto
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                export_params=True,
                do_constant_folding=True,
                verbose=False,
            )
        
        logger.info(f"Modelo exportado a ONNX: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error exportando a ONNX: {e}")
        raise


def optimize_onnx_model(
    onnx_path: Path,
    optimized_path: Optional[Path] = None,
) -> Path:
    """
    Optimiza modelo ONNX.
    
    Args:
        onnx_path: Ruta al modelo ONNX
        optimized_path: Ruta donde guardar modelo optimizado (opcional)
    
    Returns:
        Ruta del modelo optimizado
    
    Raises:
        ImportError: Si ONNX no está disponible
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX no está disponible")
    
    try:
        from onnxruntime.transformers import optimizer
        
        onnx_path = Path(onnx_path)
        if optimized_path is None:
            optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
        
        # Optimizar modelo
        optimized_model = optimizer.optimize_model(
            str(onnx_path),
            model_type='bert',  # Ajustar según tipo de modelo
            num_heads=12,
            hidden_size=768,
        )
        
        optimized_model.save_model_to_file(str(optimized_path))
        logger.info(f"Modelo ONNX optimizado guardado en: {optimized_path}")
        return optimized_path
    
    except Exception as e:
        logger.warning(f"No se pudo optimizar modelo ONNX: {e}")
        return onnx_path


def validate_onnx_model(
    onnx_path: Path,
    input_shape: Tuple[int, ...],
) -> bool:
    """
    Valida modelo ONNX.
    
    Args:
        onnx_path: Ruta al modelo ONNX
        input_shape: Shape del input (sin batch dimension)
    
    Returns:
        True si el modelo es válido
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX no disponible, no se puede validar")
        return False
    
    try:
        # Validar estructura
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # Test de inferencia
        session = ort.InferenceSession(str(onnx_path))
        dummy_input = torch.randn(1, *input_shape).numpy()
        
        outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
        
        logger.info("Modelo ONNX válido")
        return True
    
    except Exception as e:
        logger.error(f"Error validando modelo ONNX: {e}")
        return False
'''
        
        (conversion_dir / "onnx_converter.py").write_text(onnx_content, encoding="utf-8")

