"""
Model Serialization - Serialización de modelos
==============================================

Sistema para serializar y deserializar modelos de forma eficiente.
Sigue mejores prácticas de serialización.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import json
import pickle

logger = logging.getLogger(__name__)


class ModelSerializationService:
    """Servicio de serialización de modelos"""
    
    @staticmethod
    def save_model(
        model: nn.Module,
        filepath: str,
        format: str = "pth",
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Guardar modelo.
        
        Args:
            model: Modelo a guardar
            filepath: Ruta donde guardar
            format: Formato ('pth', 'pt', 'pkl', 'onnx')
            include_optimizer: Si incluir optimizer
            optimizer: Optimizador (si include_optimizer=True)
            metadata: Metadatos adicionales
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format in ["pth", "pt"]:
                # PyTorch format
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "model_config": getattr(model, "config", None),
                }
                
                if include_optimizer and optimizer:
                    save_dict["optimizer_state_dict"] = optimizer.state_dict()
                
                if metadata:
                    save_dict["metadata"] = metadata
                
                torch.save(save_dict, filepath)
                logger.info(f"Model saved to {filepath}")
            
            elif format == "pkl":
                # Pickle format
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved to {filepath} (pickle)")
            
            elif format == "onnx":
                # ONNX format
                try:
                    model.eval()
                    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust as needed
                    torch.onnx.export(
                        model,
                        dummy_input,
                        filepath,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    )
                    logger.info(f"Model saved to {filepath} (ONNX)")
                except Exception as e:
                    logger.error(f"Error exporting to ONNX: {e}")
                    return False
            
            else:
                raise ValueError(f"Unknown format: {format}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return False
    
    @staticmethod
    def load_model(
        filepath: str,
        model_class: Optional[type] = None,
        model_config: Optional[Dict[str, Any]] = None,
        map_location: Optional[str] = None,
        strict: bool = True
    ) -> Union[nn.Module, Dict[str, Any]]:
        """
        Cargar modelo.
        
        Args:
            filepath: Ruta al modelo
            model_class: Clase del modelo (para pickle)
            model_config: Configuración del modelo
            map_location: Ubicación donde mapear (None = auto)
            strict: Si cargar estrictamente (para state_dict)
        
        Returns:
            Modelo cargado o diccionario con estado
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            if filepath.suffix == ".pkl":
                # Pickle format
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model loaded from {filepath} (pickle)")
                return model
            
            elif filepath.suffix in [".pth", ".pt"]:
                # PyTorch format
                checkpoint = torch.load(filepath, map_location=map_location)
                
                if model_class:
                    # Reconstruct model
                    if model_config:
                        model = model_class(**model_config)
                    else:
                        model = model_class()
                    
                    model.load_state_dict(
                        checkpoint["model_state_dict"],
                        strict=strict
                    )
                    logger.info(f"Model loaded from {filepath}")
                    return model
                else:
                    # Return checkpoint dict
                    return checkpoint
            
            else:
                raise ValueError(f"Unknown file format: {filepath.suffix}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    @staticmethod
    def save_model_summary(
        model: nn.Module,
        filepath: str
    ) -> bool:
        """
        Guardar resumen del modelo.
        
        Args:
            model: Modelo
            filepath: Ruta donde guardar
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            summary = {
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "layers": [],
            }
            
            # Get layer information
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf module
                    layer_info = {
                        "name": name,
                        "type": type(module).__name__,
                        "parameters": sum(p.numel() for p in module.parameters()),
                    }
                    summary["layers"].append(layer_info)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Model summary saved to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model summary: {e}", exc_info=True)
            return False




