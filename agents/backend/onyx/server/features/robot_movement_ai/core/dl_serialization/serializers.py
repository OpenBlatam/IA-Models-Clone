"""
Model Serialization - Modular Serialization
===========================================

Serialización modular para modelos y datos.
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import json
import pickle

logger = logging.getLogger(__name__)


class ModelSerializer:
    """Serializador de modelos."""
    
    @staticmethod
    def save_pytorch(
        model: nn.Module,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Guardar modelo PyTorch.
        
        Args:
            model: Modelo a guardar
            path: Ruta de salida
            metadata: Metadata adicional
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_pytorch(
        path: str,
        model_class: Optional[type] = None,
        **kwargs
    ) -> nn.Module:
        """
        Cargar modelo PyTorch.
        
        Args:
            path: Ruta del modelo
            model_class: Clase del modelo (opcional)
            **kwargs: Argumentos para el modelo
            
        Returns:
            Modelo cargado
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        
        if model_class:
            model = model_class(**kwargs)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("model_class required for loading")
        
        logger.info(f"Model loaded from {path}")
        return model
    
    @staticmethod
    def save_safetensors(
        model: nn.Module,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Guardar modelo en SafeTensors.
        
        Args:
            model: Modelo a guardar
            path: Ruta de salida
            metadata: Metadata adicional
        """
        try:
            from safetensors.torch import save_file
            
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            state_dict = model.state_dict()
            save_file(state_dict, path)
            
            # Guardar metadata por separado
            if metadata:
                metadata_path = path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to SafeTensors: {path}")
        except ImportError:
            raise ImportError("safetensors not available")
    
    @staticmethod
    def load_safetensors(
        path: str,
        model: nn.Module
    ) -> nn.Module:
        """
        Cargar modelo desde SafeTensors.
        
        Args:
            path: Ruta del modelo
            model: Modelo a cargar
            
        Returns:
            Modelo con estado cargado
        """
        try:
            from safetensors.torch import load_file
            
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
            
            logger.info(f"Model loaded from SafeTensors: {path}")
            return model
        except ImportError:
            raise ImportError("safetensors not available")


class ConfigSerializer:
    """Serializador de configuración."""
    
    @staticmethod
    def save_json(config: Dict[str, Any], path: str):
        """
        Guardar configuración en JSON.
        
        Args:
            config: Configuración
            path: Ruta de salida
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """
        Cargar configuración desde JSON.
        
        Args:
            path: Ruta del archivo
            
        Returns:
            Configuración
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Config loaded from {path}")
        return config
    
    @staticmethod
    def save_pickle(obj: Any, path: str):
        """
        Guardar objeto en pickle.
        
        Args:
            obj: Objeto a guardar
            path: Ruta de salida
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        
        logger.info(f"Object saved to {path}")
    
    @staticmethod
    def load_pickle(path: str) -> Any:
        """
        Cargar objeto desde pickle.
        
        Args:
            path: Ruta del archivo
            
        Returns:
            Objeto cargado
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        
        logger.info(f"Object loaded from {path}")
        return obj








