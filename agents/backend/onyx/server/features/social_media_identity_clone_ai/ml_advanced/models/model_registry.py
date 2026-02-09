"""
Model Registry para versionado y gestión de modelos
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry de modelos para versionado"""
    
    def __init__(self, registry_path: str = "./models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Carga registry desde archivo"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": []}
    
    def _save_registry(self):
        """Guarda registry a archivo"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Registra un modelo
        
        Args:
            model_name: Nombre del modelo
            model_path: Path al modelo
            version: Versión
            metadata: Metadatos adicionales
            
        Returns:
            Model ID
        """
        model_id = f"{model_name}_v{version}"
        
        model_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "model_path": model_path,
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "is_active": True
        }
        
        self.registry["models"].append(model_entry)
        self._save_registry()
        
        logger.info(f"Modelo registrado: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de modelo"""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                return model
        return None
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Lista modelos"""
        models = self.registry["models"]
        
        if model_name:
            models = [m for m in models if m["model_name"] == model_name]
        
        if active_only:
            models = [m for m in models if m.get("is_active", True)]
        
        return sorted(models, key=lambda x: x["registered_at"], reverse=True)
    
    def load_model(self, model_id: str, device: str = "cuda") -> Optional[torch.nn.Module]:
        """Carga modelo desde registry"""
        model_info = self.get_model(model_id)
        if not model_info:
            logger.error(f"Modelo no encontrado: {model_id}")
            return None
        
        try:
            model = torch.load(model_info["model_path"], map_location=device)
            logger.info(f"Modelo cargado: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None




